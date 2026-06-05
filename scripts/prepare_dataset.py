"""Prepare code-function datasets for NLA experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.data import (  # noqa: E402
    SCHEMA_VERSION,
    build_processed_example,
    extract_code_and_description,
    get_raw_task_id,
    load_dataset_from_disk_safe,
    make_example_id,
    normalize_code_snippet,
    remove_python_comments_and_docstrings,
    rename_simple_python_identifiers,
    write_jsonl,
)


@dataclass(frozen=True)
class Candidate:
    source_dataset: str
    source_split: str
    language: str
    code: str
    reference_description: str | None
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local code datasets into NLA project-standard JSONL files."
    )
    parser.add_argument("--codesearchnet_path", default="data/raw/code_search_net_python")
    parser.add_argument("--humaneval_x_python_path", default="data/raw/humaneval_x_python")
    parser.add_argument("--humaneval_x_cpp_path", default="data/raw/humaneval_x_cpp")
    parser.add_argument("--humaneval_x_java_path", default="data/raw/humaneval_x_java")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--pilot_size", type=int, default=100)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--validation_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--max_code_chars", type=int, default=4000)
    parser.add_argument("--min_code_chars", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_missing_humaneval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def validate_sizes(args: argparse.Namespace) -> None:
    for field in ("pilot_size", "train_size", "validation_size", "test_size"):
        value = getattr(args, field)
        if value < 0:
            raise ValueError(f"{field} must be non-negative, got {value}")
    if args.min_code_chars < 0:
        raise ValueError(f"min_code_chars must be non-negative, got {args.min_code_chars}")
    if args.max_code_chars < args.min_code_chars:
        raise ValueError("max_code_chars must be greater than or equal to min_code_chars")


def dataset_is_mapping(dataset: Any) -> bool:
    try:
        keys = list(dataset.keys())
    except Exception:
        return False
    if not keys:
        return False
    try:
        return all(hasattr(dataset[key], "__iter__") for key in keys)
    except Exception:
        return False


def available_splits(dataset: Any) -> list[str]:
    if dataset_is_mapping(dataset):
        return list(dataset.keys())
    return ["all"]


def select_split(dataset: Any, preferred_names: Iterable[str]) -> tuple[Any | None, str | None]:
    if not dataset_is_mapping(dataset):
        return dataset, "all"
    for split_name in preferred_names:
        if split_name in dataset:
            return dataset[split_name], split_name
    return None, None


def iter_dataset_rows(dataset_split: Any, seed: int) -> Iterable[tuple[int, dict]]:
    rng = random.Random(seed)
    if hasattr(dataset_split, "__len__") and hasattr(dataset_split, "__getitem__"):
        indices = list(range(len(dataset_split)))
        rng.shuffle(indices)
        for index in indices:
            yield index, dict(dataset_split[index])
        return

    rows = [dict(row) for row in dataset_split]
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        yield index, row


def collect_candidates(
    dataset_split: Any,
    *,
    source_dataset: str,
    source_split: str,
    language: str,
    min_code_chars: int,
    max_code_chars: int,
    seed: int,
    max_examples: int,
    warnings: list[str],
) -> list[Candidate]:
    if max_examples <= 0:
        return []

    candidates: list[Candidate] = []
    for raw_index, row in iter_dataset_rows(dataset_split, seed):
        code, description = extract_code_and_description(row, source_dataset, language)
        code = normalize_code_snippet(code)
        if not code:
            continue
        if len(code) < min_code_chars or len(code) > max_code_chars:
            continue

        raw_task_id = get_raw_task_id(row)
        metadata = {
            "raw_index": raw_index,
            "raw_task_id": raw_task_id,
        }
        candidates.append(
            Candidate(
                source_dataset=source_dataset,
                source_split=source_split,
                language=language,
                code=code,
                reference_description=description,
                metadata=metadata,
            )
        )
        if max_examples and len(candidates) >= max_examples:
            break

    if len(candidates) < max_examples:
        warnings.append(
            f"{source_dataset}:{source_split} requested {max_examples} examples after filtering, "
            f"but only {len(candidates)} were available."
        )
    return candidates


def build_original_examples(
    candidates: list[Candidate],
    *,
    prefix: str,
    split: str,
    transformation_type: str = "original",
) -> list[dict]:
    return [
        build_processed_example(
            example_id=make_example_id(prefix, index),
            source_dataset=candidate.source_dataset,
            source_split=candidate.source_split,
            split=split,
            language=candidate.language,
            code=candidate.code,
            reference_description=candidate.reference_description,
            transformation_type=transformation_type,
            paired_example_id=None,
            metadata=candidate.metadata,
        )
        for index, candidate in enumerate(candidates)
    ]


def build_surface_shift_examples(indomain_examples: list[dict], max_examples: int) -> list[dict]:
    shifted = []
    for index, original in enumerate(indomain_examples[:max_examples]):
        transformed_code, transformation_type = transform_python_surface(original["code"])
        shifted.append(
            build_processed_example(
                example_id=make_example_id("test_surface_shift", index, transformation_type),
                source_dataset=original["source_dataset"],
                source_split=original["source_split"],
                split="test_surface_shift",
                language=original["language"],
                code=transformed_code,
                reference_description=original["reference_description"],
                transformation_type=transformation_type,
                paired_example_id=original["example_id"],
                metadata={
                    **original["metadata"],
                    "surface_shift_source_example_id": original["example_id"],
                },
            )
        )
    return shifted


def transform_python_surface(code: str) -> tuple[str, str]:
    normalized = normalize_code_snippet(code)
    renamed = rename_simple_python_identifiers(normalized)
    if renamed != normalized:
        return renamed, "rename_identifiers"

    comment_removed = remove_python_comments_and_docstrings(normalized)
    if comment_removed != normalized:
        return comment_removed, "comment_removed"

    formatting_only = "\n".join(line.rstrip() for line in normalized.splitlines()).strip()
    return formatting_only, "formatting_only"


def build_language_shift_examples(
    *,
    args: argparse.Namespace,
    warnings: list[str],
) -> list[dict]:
    paths = {
        "python": Path(args.humaneval_x_python_path),
        "cpp": Path(args.humaneval_x_cpp_path),
        "java": Path(args.humaneval_x_java_path),
    }
    missing = {language: path for language, path in paths.items() if not path.exists()}
    if missing:
        details = ", ".join(f"{language}={path}" for language, path in missing.items())
        message = f"Missing HumanEval-X dataset path(s): {details}"
        if args.allow_missing_humaneval:
            warnings.append(message + "; wrote empty test_language_shift.jsonl.")
            print(
                message
                + "; skipping language-shift test because --allow_missing_humaneval is set."
            )
            return []
        raise FileNotFoundError(
            message
            + ". Prepare local HumanEval-X datasets first; see `docs/setup_and_model_download.md`."
        )

    examples: list[dict] = []
    for language, path in paths.items():
        dataset = load_dataset_from_disk_safe(path)
        split_dataset, source_split = select_split(dataset, ("test", "validation", "train"))
        if split_dataset is None or source_split is None:
            available = ", ".join(available_splits(dataset))
            raise ValueError(f"No usable split found for {path}. Available splits: {available}")
        candidates = collect_candidates(
            split_dataset,
            source_dataset=f"humaneval_x_{language}",
            source_split=source_split,
            language=language,
            min_code_chars=args.min_code_chars,
            max_code_chars=args.max_code_chars,
            seed=args.seed + 100 + len(examples),
            max_examples=args.test_size,
            warnings=warnings,
        )
        for candidate_index, candidate in enumerate(candidates):
            raw_task_id = candidate.metadata.get("raw_task_id")
            paired_example_id = f"humanevalx_{raw_task_id}" if raw_task_id else None
            examples.append(
                build_processed_example(
                    example_id=make_example_id(
                        f"test_language_shift_{language}",
                        candidate_index,
                    ),
                    source_dataset=candidate.source_dataset,
                    source_split=candidate.source_split,
                    split="test_language_shift",
                    language=candidate.language,
                    code=candidate.code,
                    reference_description=candidate.reference_description,
                    transformation_type="language_shift",
                    paired_example_id=paired_example_id,
                    metadata=candidate.metadata,
                )
            )
    return examples


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_rows_by(rows_by_file: dict[str, list[dict]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for rows in rows_by_file.values():
        counter.update(str(row[field]) for row in rows)
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    validate_sizes(args)
    warnings: list[str] = []
    output_dir = Path(args.output_dir)

    print_section(1, 7, "Loading raw datasets")
    codesearchnet = load_dataset_from_disk_safe(Path(args.codesearchnet_path))
    print(f"CodeSearchNet available splits: {', '.join(available_splits(codesearchnet))}")

    train_split, train_source_split = select_split(codesearchnet, ("train", "training"))
    if train_split is None or train_source_split is None:
        available = ", ".join(available_splits(codesearchnet))
        raise ValueError(f"No train split found for CodeSearchNet. Available splits: {available}")

    validation_split, validation_source_split = select_split(
        codesearchnet,
        ("validation", "valid", "dev"),
    )
    test_split, test_source_split = select_split(codesearchnet, ("test", "testing"))

    print_section(2, 7, "Building train/validation/test")
    fallback_validation = validation_split is None
    fallback_test = test_split is None
    train_pool_size = args.train_size
    if fallback_validation:
        train_pool_size += args.validation_size
        warnings.append(
            "CodeSearchNet validation split missing; using deterministic train fallback."
        )
    if fallback_test:
        train_pool_size += args.test_size
        warnings.append("CodeSearchNet test split missing; using deterministic train fallback.")

    train_pool = collect_candidates(
        train_split,
        source_dataset="code_search_net_python",
        source_split=train_source_split,
        language="python",
        min_code_chars=args.min_code_chars,
        max_code_chars=args.max_code_chars,
        seed=args.seed,
        max_examples=train_pool_size,
        warnings=warnings,
    )
    train_candidates = train_pool[: args.train_size]
    offset = args.train_size

    if fallback_validation:
        validation_candidates = train_pool[offset : offset + args.validation_size]
        offset += args.validation_size
    else:
        validation_candidates = collect_candidates(
            validation_split,
            source_dataset="code_search_net_python",
            source_split=validation_source_split or "validation",
            language="python",
            min_code_chars=args.min_code_chars,
            max_code_chars=args.max_code_chars,
            seed=args.seed + 1,
            max_examples=args.validation_size,
            warnings=warnings,
        )

    if fallback_test:
        test_candidates = train_pool[offset : offset + args.test_size]
    else:
        test_candidates = collect_candidates(
            test_split,
            source_dataset="code_search_net_python",
            source_split=test_source_split or "test",
            language="python",
            min_code_chars=args.min_code_chars,
            max_code_chars=args.max_code_chars,
            seed=args.seed + 2,
            max_examples=args.test_size,
            warnings=warnings,
        )

    train_rows = build_original_examples(train_candidates, prefix="train", split="train")
    validation_rows = build_original_examples(
        validation_candidates,
        prefix="validation",
        split="validation",
    )
    test_indomain_rows = build_original_examples(
        test_candidates,
        prefix="test_indomain",
        split="test_indomain",
    )

    print_section(3, 7, "Building pilot")
    pilot_candidates = train_candidates[: args.pilot_size]
    if len(pilot_candidates) < args.pilot_size:
        warnings.append(
            f"Requested pilot_size={args.pilot_size}, but only {len(pilot_candidates)} "
            "training examples were available."
        )
    pilot_rows = build_original_examples(pilot_candidates, prefix="pilot", split="pilot")

    print_section(4, 7, "Building surface-shift test")
    test_surface_shift_rows = build_surface_shift_examples(test_indomain_rows, args.test_size)

    print_section(5, 7, "Building language-shift test")
    test_language_shift_rows = build_language_shift_examples(args=args, warnings=warnings)

    print_section(6, 7, "Writing outputs")
    rows_by_file = {
        "pilot_100.jsonl": pilot_rows,
        "train.jsonl": train_rows,
        "validation.jsonl": validation_rows,
        "test_indomain.jsonl": test_indomain_rows,
        "test_surface_shift.jsonl": test_surface_shift_rows,
        "test_language_shift.jsonl": test_language_shift_rows,
    }
    output_counts = {}
    for filename, rows in rows_by_file.items():
        count = write_jsonl(output_dir / filename, rows)
        output_counts[filename] = count
        print(f"Wrote {filename}: {count} rows")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "input_paths": {
            "codesearchnet_path": args.codesearchnet_path,
            "humaneval_x_python_path": args.humaneval_x_python_path,
            "humaneval_x_cpp_path": args.humaneval_x_cpp_path,
            "humaneval_x_java_path": args.humaneval_x_java_path,
        },
        "output_counts": output_counts,
        "counts_by_language": count_rows_by(rows_by_file, "language"),
        "counts_by_transformation_type": count_rows_by(rows_by_file, "transformation_type"),
        "warnings": warnings,
    }
    write_manifest(output_dir / "dataset_manifest.json", manifest)
    print("Wrote dataset_manifest.json")

    print_section(7, 7, "Summary")
    for filename, count in output_counts.items():
        print(f"{filename}: {count}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print("SUCCESS: Phase 2 dataset preparation completed successfully.")


if __name__ == "__main__":
    main()
