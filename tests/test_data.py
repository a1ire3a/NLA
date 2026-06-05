from __future__ import annotations

from pathlib import Path

import pytest

from nla_code_interp.data import (
    TASK_FAMILY,
    build_processed_example,
    iter_jsonl,
    load_dataset_from_disk_safe,
    normalize_code_snippet,
    rename_simple_python_identifiers,
    write_jsonl,
)


def test_jsonl_write_read_roundtrip(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"b": "two"}]
    path = tmp_path / "rows.jsonl"

    count = write_jsonl(path, rows)

    assert count == 2
    assert list(iter_jsonl(path)) == rows


def test_build_processed_example_schema_and_prompt() -> None:
    code = "def add(a, b):\n    return a + b"

    row = build_processed_example(
        example_id="train_000001",
        source_dataset="synthetic",
        source_split="train",
        split="train",
        language="python",
        code=code,
        reference_description="Add two values.",
        transformation_type="original",
        metadata={"raw_index": 0},
    )

    assert list(row) == [
        "example_id",
        "source_dataset",
        "source_split",
        "split",
        "language",
        "task_family",
        "code",
        "prompt",
        "reference_description",
        "transformation_type",
        "paired_example_id",
        "metadata",
    ]
    assert row["task_family"] == TASK_FAMILY
    assert row["code"] == code
    assert "<code>\n" in row["prompt"]
    assert code in row["prompt"]
    assert row["paired_example_id"] is None


def test_normalize_code_snippet_line_endings() -> None:
    code = "\r\ndef add(a, b):\r\n    return a + b\r\n"

    assert normalize_code_snippet(code) == "def add(a, b):\n    return a + b"


def test_rename_simple_python_identifiers_preserves_keywords_and_builtins() -> None:
    code = "def add(value):\n    if value > 0:\n        return len([value])\n    return 0"

    renamed = rename_simple_python_identifiers(code)

    assert "def " in renamed
    assert "if " in renamed
    assert "return " in renamed
    assert "len" in renamed
    assert "value" not in renamed


def test_load_dataset_from_disk_safe_missing_path_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_dataset"

    with pytest.raises(FileNotFoundError, match="docs/setup_and_model_download.md"):
        load_dataset_from_disk_safe(missing_path)
