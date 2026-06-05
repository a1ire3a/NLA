from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nla_code_interp.data import write_jsonl
from scripts.run_evaluation import load_activation_artifact


def write_fake_artifact(
    path: Path,
    activations: torch.Tensor,
    metadata_rows: list[dict],
    manifest: dict,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(activations, path / "activations.pt")
    write_jsonl(path / "metadata.jsonl", metadata_rows)
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def test_load_activation_artifact_success(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metadata_rows = [
        {"activation_index": 0, "example_id": "ex_0"},
        {"activation_index": 1, "example_id": "ex_1"},
    ]
    manifest = {"num_examples": 2, "activation_dim": 2}
    write_fake_artifact(artifact_dir, activations, metadata_rows, manifest)

    artifact = load_activation_artifact(artifact_dir)

    assert artifact.activations.shape == (2, 2)
    assert len(artifact.metadata_rows) == 2


def test_load_activation_artifact_metadata_row_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metadata_rows = [{"activation_index": 0, "example_id": "ex_0"}]
    manifest = {"num_examples": 2, "activation_dim": 2}
    write_fake_artifact(artifact_dir, activations, metadata_rows, manifest)

    with pytest.raises(ValueError, match="Metadata row count"):
        load_activation_artifact(artifact_dir)


def test_load_activation_artifact_manifest_dim_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metadata_rows = [
        {"activation_index": 0, "example_id": "ex_0"},
        {"activation_index": 1, "example_id": "ex_1"},
    ]
    manifest = {"num_examples": 2, "activation_dim": 3}
    write_fake_artifact(artifact_dir, activations, metadata_rows, manifest)

    with pytest.raises(ValueError, match="Manifest activation_dim"):
        load_activation_artifact(artifact_dir)


def test_load_activation_artifact_nonsequential_indices(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metadata_rows = [
        {"activation_index": 1, "example_id": "ex_0"},
        {"activation_index": 2, "example_id": "ex_1"},
    ]
    manifest = {"num_examples": 2, "activation_dim": 2}
    write_fake_artifact(artifact_dir, activations, metadata_rows, manifest)

    with pytest.raises(ValueError, match="not sequential"):
        load_activation_artifact(artifact_dir)
