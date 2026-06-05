from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.run_qwen_nla_loop import (
    build_reconstruction_methods,
    metric_rows_for_methods,
    output_paths,
    per_example_metric_rows,
    prepare_output_paths,
    subset_artifact,
)
from scripts.train_ar import ActivationArtifact
from scripts.train_qwen_nla_reconstruction import (
    build_generated_text_examples,
    build_manifest_payload,
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
        }
        for index in range(num_examples)
    ]
    return ActivationArtifact(
        activation_dir=Path("fake"),
        activations=activations,
        metadata_rows=metadata_rows,
        manifest={"num_examples": num_examples, "activation_dim": activation_dim},
    )


def fake_generated_rows(count: int) -> list[dict]:
    return [
        {
            "activation_index": index,
            "example_id": f"ex_{index}",
            "target_text": f"target {index}",
            "generated_text": f"generated explanation {index}",
            "split": "validation",
            "language": "python",
            "transformation_type": "original",
        }
        for index in range(count)
    ]


def test_qwen_nla_output_paths_and_overwrite_refusal(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "run_a")

    assert paths["metrics_csv"].name == "run_a_qwen_nla_metrics.csv"
    assert paths["metrics_json"].name == "run_a_qwen_nla_metrics.json"
    assert paths["generated"].name == "run_a_generated_explanations.jsonl"
    assert paths["per_example"].name == "run_a_per_example_metrics.jsonl"
    assert paths["manifest"].name == "run_a_manifest.json"

    prepare_output_paths(paths, overwrite=False)
    paths["metrics_json"].write_text("[]\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        prepare_output_paths(paths, overwrite=False)

    prepare_output_paths(paths, overwrite=True)
    assert not paths["metrics_json"].exists()


def test_limit_handling_preserves_first_rows() -> None:
    artifact = fake_artifact(num_examples=4, activation_dim=3)

    subset = subset_artifact(artifact, 2)

    assert subset.activations.shape == (2, 3)
    assert [row["activation_index"] for row in subset.metadata_rows] == [0, 1]


def test_metrics_include_qwen_nla_and_baselines() -> None:
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    reconstruction = original.clone()

    reconstructions = build_reconstruction_methods(
        original=original,
        qwen_nla_reconstruction=reconstruction,
        seed=42,
    )
    rows = metric_rows_for_methods(
        run_name="run_a",
        original=original,
        reconstructions=reconstructions,
    )

    assert set(reconstructions) == {"qwen_nla", "mean", "zero", "shuffled"}
    assert all(value.shape == original.shape for value in reconstructions.values())
    assert [row["method"] for row in rows] == ["qwen_nla", "mean", "zero", "shuffled"]
    assert rows[0]["fve"] == pytest.approx(1.0)


def test_per_example_metric_row_schema() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=2)
    generated_rows = fake_generated_rows(2)

    rows = per_example_metric_rows(
        metadata_rows=artifact.metadata_rows,
        original=artifact.activations,
        reconstructed=artifact.activations.clone(),
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
    assert rows[0]["generated_text"] == "generated explanation 0"


def test_generated_text_adaptation_dataset_uses_generated_text() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=2)
    generated_rows = fake_generated_rows(2)
    transformed_targets = torch.zeros((2, 2))

    examples = build_generated_text_examples(
        artifact=artifact,
        generated_rows=generated_rows,
        transformed_targets=transformed_targets,
    )

    assert len(examples) == 2
    assert examples[0].text == "generated explanation 0"
    assert examples[0].metadata["qwen_nla_source_text"] == "generated explanation 0"
    assert torch.equal(examples[0].target, torch.zeros(2))


def test_generated_text_dataset_rejects_count_mismatch() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=2)

    with pytest.raises(ValueError, match="Generated row count"):
        build_generated_text_examples(
            artifact=artifact,
            generated_rows=fake_generated_rows(1),
            transformed_targets=torch.zeros((2, 2)),
        )


def test_reconstruction_adaptation_manifest_schema() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=2)

    class Transform:
        name = "standardize"

        def state_dict_for_manifest(self) -> dict:
            return {"name": self.name, "mean_shape": [1, 2], "std_shape": [1, 2]}

    manifest = build_manifest_payload(
        args={
            "activation_dir": "train_dir",
            "validation_activation_dir": "validation_dir",
            "qwen_av_checkpoint_dir": "av_dir",
            "qwen_ar_checkpoint_dir": "ar_dir",
            "output_dir": "out_dir",
            "limit_train": 64,
            "limit_validation": 32,
        },
        train_artifact=artifact,
        validation_artifact=artifact,
        av_checkpoint_summary={"config": {"component": "qwen_av"}},
        ar_checkpoint_summary={"config": {"component": "qwen_ar"}},
        target_transform=Transform(),
        best_epoch=1,
        best_metrics={"validation_fve": 0.1},
        output_files={
            "train_generated_explanations": "train_generated_explanations.jsonl",
            "validation_generated_explanations": "validation_generated_explanations.jsonl",
        },
    )

    assert manifest["schema_version"] == "phase10c_qwen_nla_reconstruction_v1"
    assert manifest["qwen_av_checkpoint_dir"] == "av_dir"
    assert manifest["qwen_ar_checkpoint_dir"] == "ar_dir"
    assert manifest["limit_train"] == 64
    assert manifest["target_transform"]["name"] == "standardize"
    assert "train_generated_explanations" in manifest["output_files"]
