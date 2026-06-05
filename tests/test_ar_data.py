from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from scripts.train_ar import (
    ActivationArtifact,
    TargetTransform,
    build_text_examples,
    build_per_example_validation_metrics,
    build_train_validation_data,
    resolve_target_transform_arg,
    select_text_for_row,
    split_train_validation_indices,
    text_field_counts,
    text_length_summary,
    validation_train_mean_baseline_metrics,
)


def fake_artifact(name: str, activations: torch.Tensor, split: str) -> ActivationArtifact:
    rows = [
        {
            "activation_index": index,
            "example_id": f"{name}_{index}",
            "split": split,
            "reference_description": f"description {index}",
            "prompt": f"prompt {index}",
            "code": f"code {index}",
        }
        for index in range(activations.shape[0])
    ]
    return ActivationArtifact(
        activation_dir=Path(name),
        activations=activations,
        metadata_rows=rows,
        manifest={"num_examples": activations.shape[0], "activation_dim": activations.shape[1]},
    )


def test_text_fallback_uses_reference_description_first() -> None:
    row = {
        "example_id": "ex_0",
        "reference_description": " reference text ",
        "prompt": "prompt text",
        "code": "code text",
    }

    text, selected_field = select_text_for_row(
        row,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert text == "reference text"
    assert selected_field == "reference_description"


def test_text_fallback_uses_prompt_then_code() -> None:
    prompt_row = {
        "example_id": "ex_1",
        "reference_description": None,
        "prompt": "prompt text",
        "code": "code text",
    }
    code_row = {
        "example_id": "ex_2",
        "reference_description": "",
        "prompt": " ",
        "code": "code text",
    }

    prompt_text, prompt_field = select_text_for_row(
        prompt_row,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )
    code_text, code_field = select_text_for_row(
        code_row,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert prompt_text == "prompt text"
    assert prompt_field == "prompt"
    assert code_text == "code text"
    assert code_field == "code"


def test_build_text_examples_records_selected_field() -> None:
    rows = [
        {
            "example_id": "ex_0",
            "reference_description": "",
            "prompt": "prompt text",
            "code": "code text",
        }
    ]

    examples = build_text_examples(
        rows,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert examples[0].text == "prompt text"
    assert examples[0].metadata["ar_text_field"] == "prompt"
    assert examples[0].metadata["ar_source_index"] == 0


def test_deterministic_train_validation_split_is_stable() -> None:
    rows = [{"split": "pilot", "example_id": f"ex_{index}"} for index in range(10)]

    train_1, validation_1, strategy_1 = split_train_validation_indices(
        rows,
        validation_fraction=0.2,
        seed=42,
    )
    train_2, validation_2, strategy_2 = split_train_validation_indices(
        rows,
        validation_fraction=0.2,
        seed=42,
    )

    assert strategy_1 == "deterministic_random_split"
    assert strategy_2 == "deterministic_random_split"
    assert train_1 == train_2
    assert validation_1 == validation_2
    assert len(train_1) == 8
    assert len(validation_1) == 2
    assert set(train_1).isdisjoint(validation_1)


def test_explicit_train_validation_split_is_respected() -> None:
    rows = [
        {"split": "train", "example_id": "ex_0"},
        {"split": "validation", "example_id": "ex_1"},
        {"split": "train", "example_id": "ex_2"},
    ]

    train, validation, strategy = split_train_validation_indices(
        rows,
        validation_fraction=0.2,
        seed=42,
    )

    assert strategy == "metadata_split"
    assert train == [0, 2]
    assert validation == [1]


def test_build_train_validation_data_internal_split_unchanged() -> None:
    artifact = fake_artifact("train", torch.zeros((10, 3)), "pilot")

    data = build_train_validation_data(
        train_artifact=artifact,
        validation_artifact=None,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
        validation_fraction=0.2,
        seed=42,
    )

    assert data.uses_external_validation is False
    assert data.train_artifact is artifact
    assert data.validation_artifact is artifact
    assert data.split_strategy == "deterministic_random_split"
    assert len(data.train_indices) == 8
    assert len(data.validation_indices) == 2


def test_build_train_validation_data_external_validation_uses_all_rows() -> None:
    train_artifact = fake_artifact("train", torch.zeros((4, 3)), "train")
    validation_artifact = fake_artifact("validation", torch.zeros((2, 3)), "validation")

    data = build_train_validation_data(
        train_artifact=train_artifact,
        validation_artifact=validation_artifact,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
        validation_fraction=0.2,
        seed=42,
    )

    assert data.uses_external_validation is True
    assert data.split_strategy == "external_validation_artifact"
    assert data.train_indices == [0, 1, 2, 3]
    assert data.validation_indices == [0, 1]
    assert data.validation_artifact is validation_artifact
    assert data.validation_examples[0].metadata["example_id"] == "validation_0"


def test_build_train_validation_data_external_dim_mismatch_error() -> None:
    train_artifact = fake_artifact("train", torch.zeros((4, 3)), "train")
    validation_artifact = fake_artifact("validation", torch.zeros((2, 4)), "validation")

    with pytest.raises(ValueError, match="activation dim does not match"):
        build_train_validation_data(
            train_artifact=train_artifact,
            validation_artifact=validation_artifact,
            text_field="reference_description",
            fallback_text_fields=["prompt", "code"],
            validation_fraction=0.2,
            seed=42,
        )


def test_target_transform_raw_identity() -> None:
    targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    transform = TargetTransform.fit("raw", targets)

    transformed = transform.transform(targets)
    restored = transform.inverse_transform(transformed)

    assert torch.equal(transformed, targets)
    assert torch.equal(restored, targets)
    assert transform.state_dict_for_manifest()["name"] == "raw"


def test_target_transform_center_uses_train_mean() -> None:
    train_targets = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    validation_targets = torch.tensor([[5.0, 10.0]])
    transform = TargetTransform.fit("center", train_targets)

    transformed = transform.transform(validation_targets)
    restored = transform.inverse_transform(transformed)

    assert torch.allclose(transformed, torch.tensor([[3.0, 6.0]]))
    assert torch.allclose(restored, validation_targets)
    assert transform.state_dict_for_manifest()["mean"] == [2.0, 4.0]


def test_target_transform_standardize_clamps_zero_std() -> None:
    train_targets = torch.tensor([[1.0, 2.0], [1.0, 4.0]])
    validation_targets = torch.tensor([[1.0, 6.0]])
    transform = TargetTransform.fit("standardize", train_targets, eps=1e-6)

    transformed = transform.transform(validation_targets)
    restored = transform.inverse_transform(transformed)

    assert transformed[0, 0].item() == pytest.approx(0.0)
    assert transformed[0, 1].item() == pytest.approx(3.0)
    assert torch.allclose(restored, validation_targets)
    assert transform.std is not None
    assert transform.std[0, 0].item() == pytest.approx(1e-6)


def test_predict_residual_from_mean_resolves_to_center() -> None:
    args = argparse.Namespace(
        target_transform=None,
        predict_residual_from_mean=True,
    )

    resolved = resolve_target_transform_arg(args)

    assert resolved == "center"
    assert args.target_transform == "center"


def test_predict_residual_from_mean_rejects_conflicting_transform() -> None:
    args = argparse.Namespace(
        target_transform="standardize",
        predict_residual_from_mean=True,
    )

    with pytest.raises(ValueError, match="alias for --target_transform center"):
        resolve_target_transform_arg(args)


def test_validation_train_mean_baseline_uses_train_targets_only() -> None:
    train_targets = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    validation_targets = torch.tensor([[10.0, 0.0]])

    metrics = validation_train_mean_baseline_metrics(
        train_targets=train_targets,
        validation_targets=validation_targets,
    )

    assert metrics["validation_train_mean_baseline_mse"] == pytest.approx(40.5)


def test_per_example_validation_metrics_required_fields() -> None:
    metadata_rows = [
        {"activation_index": 0, "example_id": "ex_0", "ar_text_field": "prompt"},
        {"activation_index": 1, "example_id": "ex_1", "ar_text_field": "code"},
    ]
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    predictions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    rows = build_per_example_validation_metrics(
        metadata_rows=metadata_rows,
        original_targets=targets,
        original_predictions=predictions,
    )

    assert len(rows) == 2
    assert set(rows[0]) == {
        "activation_index",
        "example_id",
        "selected_text_field",
        "squared_error",
        "l2_error",
        "cosine_similarity",
        "target_norm",
        "prediction_norm",
    }


def test_text_field_counts_and_length_summary() -> None:
    rows = [
        {
            "example_id": "ex_0",
            "reference_description": "short",
            "prompt": "prompt text",
            "code": "code text",
        },
        {
            "example_id": "ex_1",
            "reference_description": "",
            "prompt": "longer prompt",
            "code": "code text",
        },
    ]
    examples = build_text_examples(
        rows,
        text_field="reference_description",
        fallback_text_fields=["prompt", "code"],
    )

    assert text_field_counts(examples) == {"prompt": 1, "reference_description": 1}
    summary = text_length_summary(examples)
    assert summary["count"] == 2
    assert summary["max_chars"] == len("longer prompt")
