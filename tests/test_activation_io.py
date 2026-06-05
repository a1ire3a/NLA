from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nla_code_interp.activations import activation_save_dtype, summarize_activation_batch
from nla_code_interp.data import iter_jsonl, write_jsonl


def test_activation_save_dtype_mapping() -> None:
    assert activation_save_dtype("float32") == torch.float32
    assert activation_save_dtype("float16") == torch.float16
    assert activation_save_dtype("bfloat16") == torch.bfloat16


def test_activation_save_dtype_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported activation save dtype"):
        activation_save_dtype("int8")


def test_summarize_activation_batch_keys_and_values() -> None:
    activations = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    summary = summarize_activation_batch(activations)

    assert set(summary) == {"mean", "std", "min", "max", "average_l2_norm"}
    assert summary["mean"] == pytest.approx(1.75)
    assert summary["min"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(4.0)
    assert summary["average_l2_norm"] == pytest.approx(2.5)


def test_fake_activation_artifact_row_count_matches_metadata(tmp_path: Path) -> None:
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor_path = tmp_path / "activations.pt"
    metadata_path = tmp_path / "metadata.jsonl"
    torch.save(activations, tensor_path)
    write_jsonl(
        metadata_path,
        [
            {"activation_index": 0, "example_id": "ex_0"},
            {"activation_index": 1, "example_id": "ex_1"},
        ],
    )

    loaded = torch.load(tensor_path, map_location="cpu")
    metadata_rows = list(iter_jsonl(metadata_path))

    assert loaded.shape[0] == len(metadata_rows)
    assert loaded.shape == (2, 2)
