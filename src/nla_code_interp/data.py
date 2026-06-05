"""Dataset preparation utilities for code-level NLA experiments."""

from __future__ import annotations

import ast
import builtins
import io
import json
import keyword
import re
import token
import tokenize
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from nla_code_interp.prompts import code_explanation_prompt_v1


TASK_FAMILY = "function_level_code_understanding"
SCHEMA_VERSION = "phase2_v1"
TRANSFORMATION_TYPES = {
    "original",
    "rename_identifiers",
    "formatting_only",
    "comment_removed",
    "language_shift",
}

CODE_FIELD_CANDIDATES = (
    "func_code_string",
    "code",
    "function",
    "original_string",
    "declaration",
    "solution",
    "completion",
)
DESCRIPTION_FIELD_CANDIDATES = (
    "func_documentation_string",
    "docstring",
    "doc",
    "description",
    "text",
    "instruction",
    "prompt",
    "canonical_solution",
)
TASK_ID_FIELD_CANDIDATES = ("task_id", "task_name", "problem_id", "id", "name")


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield dictionaries from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object in {path} at line {line_number}, got {type(row)}"
                )
            yield row


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Write rows to a JSONL file and return the number of rows written."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_code_snippet(code: str) -> str:
    """Conservatively normalize a code snippet for prompt construction."""
    if not isinstance(code, str):
        raise TypeError(f"code must be a string, got {type(code)}")
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()


def make_example_id(prefix: str, index: int, variant: str | None = None) -> str:
    """Create a deterministic example id."""
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")
    parts = [_sanitize_id_part(prefix), f"{index:06d}"]
    if variant is not None:
        parts.append(_sanitize_id_part(variant))
    return "_".join(parts)


def build_processed_example(
    *,
    example_id: str,
    source_dataset: str,
    source_split: str,
    split: str,
    language: str,
    code: str,
    reference_description: str | None,
    transformation_type: str,
    paired_example_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Build and validate one processed dataset row."""
    required = {
        "example_id": example_id,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "split": split,
        "language": language,
    }
    for field_name, value in required.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if transformation_type not in TRANSFORMATION_TYPES:
        allowed = ", ".join(sorted(TRANSFORMATION_TYPES))
        raise ValueError(f"Invalid transformation_type={transformation_type!r}; allowed: {allowed}")

    normalized_code = normalize_code_snippet(code)
    if not normalized_code:
        raise ValueError("code must be non-empty after normalization")
    if reference_description is not None and not isinstance(reference_description, str):
        raise TypeError("reference_description must be a string or None")
    if paired_example_id is not None and not isinstance(paired_example_id, str):
        raise TypeError("paired_example_id must be a string or None")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict or None")

    return {
        "example_id": example_id,
        "source_dataset": source_dataset.strip(),
        "source_split": source_split.strip(),
        "split": split.strip(),
        "language": language.strip(),
        "task_family": TASK_FAMILY,
        "code": normalized_code,
        "prompt": code_explanation_prompt_v1(normalized_code),
        "reference_description": reference_description.strip() if reference_description else None,
        "transformation_type": transformation_type,
        "paired_example_id": paired_example_id,
        "metadata": dict(metadata or {}),
    }


def load_dataset_from_disk_safe(path: Path):
    """Load a local Hugging Face dataset with a clear missing-path error."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_path}. Expected a local dataset created with "
            "`datasets.save_to_disk` or equivalent. Download/prepare the raw dataset first; see "
            "`docs/setup_and_model_download.md`."
        )
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "Could not import `datasets`. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return load_from_disk(str(dataset_path))


def extract_code_and_description(
    row: dict,
    source_dataset: str,
    language: str,
) -> tuple[str, str | None]:
    """Extract code and an optional reference description from a raw dataset row."""
    if not isinstance(row, dict):
        row = dict(row)

    code = _first_text_field(row, CODE_FIELD_CANDIDATES)
    prompt = _text_or_none(row.get("prompt"))
    canonical_solution = _text_or_none(row.get("canonical_solution"))
    if code is None and prompt and canonical_solution:
        code = prompt.rstrip() + "\n" + canonical_solution.lstrip()
    elif code is None and canonical_solution and _looks_like_code(canonical_solution, language):
        code = canonical_solution
    elif code is None and prompt and _looks_like_code(prompt, language):
        code = prompt

    description = _first_text_field(row, DESCRIPTION_FIELD_CANDIDATES)
    if description == code:
        description = None

    return normalize_code_snippet(code or ""), description.strip() if description else None


def remove_python_comments_and_docstrings(code: str) -> str:
    """Remove Python comments and obvious docstrings without trying to reformat code."""
    normalized = normalize_code_snippet(code)
    try:
        tree = ast.parse(normalized)
        docstring_spans = _collect_docstring_spans(tree)
    except SyntaxError:
        docstring_spans = set()

    try:
        tokens = tokenize.generate_tokens(io.StringIO(normalized).readline)
        kept = []
        for token_info in tokens:
            if token_info.type == tokenize.COMMENT:
                continue
            if token_info.type == token.STRING and _span_contains_token(
                docstring_spans,
                token_info.start,
                token_info.end,
            ):
                continue
            kept.append(token_info)
        return normalize_code_snippet(tokenize.untokenize(kept))
    except tokenize.TokenError:
        return normalized


def rename_simple_python_identifiers(code: str, prefix: str = "var") -> str:
    """Conservatively rename Python identifiers using tokenizer-level replacement.

    The function avoids keywords, builtins, strings, comments, imports, and attribute
    names. It is intentionally conservative and does not guarantee semantic equivalence
    for every Python program.
    """
    normalized = normalize_code_snippet(code)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(normalized).readline))
    except tokenize.TokenError:
        return normalized

    prefix = _sanitize_id_part(prefix)
    import_lines, imported_names = _collect_import_context(tokens)
    existing_names = {token_info.string for token_info in tokens if token_info.type == token.NAME}
    excluded_names = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}
    excluded_names |= imported_names

    mapping: dict[str, str] = {}
    for index, token_info in enumerate(tokens):
        if token_info.type != token.NAME:
            continue
        name = token_info.string
        if name in excluded_names or token_info.start[0] in import_lines:
            continue
        previous = _previous_significant_token(tokens, index)
        if previous is not None and previous.string == ".":
            continue
        if name not in mapping:
            unavailable = existing_names | excluded_names | set(mapping.values())
            mapping[name] = _next_rename(prefix, unavailable)

    if not mapping:
        return normalized

    renamed_tokens = []
    for token_info in tokens:
        if token_info.type == token.NAME and token_info.string in mapping:
            renamed_tokens.append(token_info._replace(string=mapping[token_info.string]))
        else:
            renamed_tokens.append(token_info)
    return normalize_code_snippet(tokenize.untokenize(renamed_tokens))


def get_raw_task_id(row: dict) -> str | None:
    """Return a common raw task id if present."""
    value = _first_text_field(row, TASK_ID_FIELD_CANDIDATES)
    return value.strip() if value else None


def _sanitize_id_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return sanitized or "example"


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _first_text_field(row: dict, field_names: Iterable[str]) -> str | None:
    for field_name in field_names:
        text = _text_or_none(row.get(field_name))
        if text:
            return text
    return None


def _looks_like_code(text: str, language: str) -> bool:
    markers = ("def ", "class ", "return ", "{", "}", ";", "#include", "public static")
    if language.lower() == "python":
        markers = ("def ", "class ", "return ", "import ", "from ")
    return any(marker in text for marker in markers)


def _collect_docstring_spans(tree: ast.AST) -> set[tuple[int, int, int, int]]:
    spans: set[tuple[int, int, int, int]] = set()
    for node in [tree, *ast.walk(tree)]:
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        value = getattr(first, "value", None)
        if (
            isinstance(first, ast.Expr)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            end_lineno = getattr(first, "end_lineno", first.lineno)
            end_col = getattr(first, "end_col_offset", first.col_offset)
            spans.add((first.lineno, first.col_offset, end_lineno, end_col))
    return spans


def _span_contains_token(
    spans: set[tuple[int, int, int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    for start_line, start_col, end_line, end_col in spans:
        starts_inside = start > (start_line, start_col) or start == (start_line, start_col)
        ends_inside = end < (end_line, end_col) or end == (end_line, end_col)
        if starts_inside and ends_inside:
            return True
    return False


def _collect_import_context(tokens: list[tokenize.TokenInfo]) -> tuple[set[int], set[str]]:
    import_lines = set()
    imported_names = set()
    for token_info in tokens:
        if token_info.type != token.NAME or token_info.string not in {"import", "from"}:
            continue
        import_lines.add(token_info.start[0])

    for token_info in tokens:
        if token_info.start[0] not in import_lines or token_info.type != token.NAME:
            continue
        if token_info.string not in {"import", "from", "as"}:
            imported_names.add(token_info.string)
    return import_lines, imported_names


def _next_rename(prefix: str, unavailable: set[str]) -> str:
    index = 0
    while True:
        candidate = f"{prefix}_{index}"
        if candidate not in unavailable:
            return candidate
        index += 1


def _previous_significant_token(
    tokens: list[tokenize.TokenInfo],
    index: int,
) -> tokenize.TokenInfo | None:
    ignored = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
    for previous_index in range(index - 1, -1, -1):
        previous = tokens[previous_index]
        if previous.type not in ignored:
            return previous
    return None
