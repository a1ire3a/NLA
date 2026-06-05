from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.run_evaluation import ActivationArtifact
from scripts.run_nla_loop import (
    build_reconstruction_methods,
    generated_explanation_row,
    metric_rows_for_methods,
    output_paths,
    per_example_nla_metric_rows,
    prepare_output_dir,
    subset_artifact,
    target_transform_from_checkpoint_state,
)


def fake_artifact(num_examples: int = 4, activation_dim: int = 3) -> ActivationArtifact:
    activations = torch.arange(num_examples * activation_dim, dtype=torch.float32).reshape(
        num_examples,
        activation_dim,
    )
    metadata_rows = [
        {
            "activation_index": index,
            "example_id": f"ex_{index}",
            "split": "validation",
            "language": "python",
            "transformation_type": "original",
            "reference_description": f"description {index}",
        }
        for index in range(num_examples)
    ]
    return ActivationArtifact(
        activation_dir=Path("fake_artifact"),
        activations=activations,
        metadata_rows=metadata_rows,
        manifest={"num_examples": num_examples, "activation_dim": activation_dim},
    )


def test_output_paths_and_overwrite_refusal(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "run_a")

    assert paths["metrics_csv"].name == "run_a_nla_metrics.csv"
    assert paths["metrics_json"].name == "run_a_nla_metrics.json"
    assert paths["generated"].name == "run_a_generated_explanations.jsonl"
    assert paths["per_example"].name == "run_a_per_example_metrics.jsonl"
    assert paths["manifest"].name == "run_a_manifest.json"

    prepare_output_dir(tmp_path, paths, overwrite=False)
    paths["metrics_json"].write_text("[]\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        prepare_output_dir(tmp_path, paths, overwrite=False)

    prepare_output_dir(tmp_path, paths, overwrite=True)
    assert not paths["metrics_json"].exists()


def test_subset_artifact_limit_preserves_first_rows() -> None:
    artifact = fake_artifact(num_examples=4, activation_dim=3)

    subset = subset_artifact(artifact, 2)

    assert subset.activations.shape == (2, 3)
    assert subset.metadata_rows[0]["example_id"] == "ex_0"
    assert subset.metadata_rows[1]["example_id"] == "ex_1"
    assert subset.metadata_rows[1]["activation_index"] == 1


def test_subset_artifact_limit_cannot_exceed_available_rows() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=3)

    with pytest.raises(ValueError, match="exceeds available examples"):
        subset_artifact(artifact, 3)


def test_generated_explanation_row_schema() -> None:
    row = generated_explanation_row(
        source_index=0,
        metadata={
            "activation_index": 7,
            "example_id": "ex_7",
            "reference_description": "reference",
            "split": "validation",
            "language": "python",
            "transformation_type": "original",
        },
        target_text="target",
        generated_text=" generated text ",
    )

    assert set(row) == {
        "activation_index",
        "example_id",
        "target_text",
        "reference_description",
        "generated_text",
        "split",
        "language",
        "transformation_type",
    }
    assert row["activation_index"] == 7
    assert row["target_text"] == "target"
    assert row["reference_description"] == "reference"
    assert row["generated_text"] == "generated text"


def test_per_example_metric_row_schema() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=2)
    reconstructed = artifact.activations.clone()
    generated_rows = [
        {"generated_text": "generated 0"},
        {"generated_text": "generated 1"},
    ]

    rows = per_example_nla_metric_rows(
        metadata_rows=artifact.metadata_rows,
        original=artifact.activations,
        reconstructed=reconstructed,
        generated_rows=generated_rows,
    )

    assert len(rows) == 2
    assert set(rows[0]) == {
        "activation_index",
        "example_id",
        "squared_error",
        "l2_error",
        "cosine_similarity",
        "generated_text",
    }
    assert rows[0]["activation_index"] == 0
    assert rows[0]["generated_text"] == "generated 0"
    assert rows[0]["squared_error"] == pytest.approx(0.0)


def test_metric_rows_include_nla_and_baselines_with_matching_shapes() -> None:
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    nla_reconstruction = original.clone()

    reconstructions = build_reconstruction_methods(
        original=original,
        nla_reconstruction=nla_reconstruction,
        seed=42,
    )
    rows = metric_rows_for_methods(
        run_name="run_a",
        original=original,
        reconstructions=reconstructions,
    )

    assert set(reconstructions) == {"nla", "mean", "zero", "shuffled"}
    assert all(tensor.shape == original.shape for tensor in reconstructions.values())
    assert [row["method"] for row in rows] == ["nla", "mean", "zero", "shuffled"]
    assert rows[0]["fve"] == pytest.approx(1.0)


def test_target_transform_from_checkpoint_state_raw_center_standardize() -> None:
    predictions = torch.tensor([[1.0, 2.0]])

    raw = target_transform_from_checkpoint_state({"name": "raw"})
    assert torch.equal(raw.inverse_transform(predictions), predictions)

    center = target_transform_from_checkpoint_state(
        {
            "name": "center",
            "mean": torch.tensor([[10.0, 20.0]]),
        }
    )
    assert torch.equal(
        center.inverse_transform(torch.zeros((1, 2))),
        torch.tensor([[10.0, 20.0]]),
    )

    standardize = target_transform_from_checkpoint_state(
        {
            "name": "standardize",
            "mean": [10.0, 20.0],
            "std": [2.0, 4.0],
        }
    )
    assert torch.equal(
        standardize.inverse_transform(torch.ones((1, 2))),
        torch.tensor([[12.0, 24.0]]),
    )


def test_target_transform_from_checkpoint_state_requires_standardize_std() -> None:
    with pytest.raises(ValueError, match="missing std"):
        target_transform_from_checkpoint_state(
            {
                "name": "standardize",
                "mean": [10.0, 20.0],
            }
        )
