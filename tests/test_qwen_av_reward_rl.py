from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.run_qwen_nla_loop import resolve_checkpoint_mode as resolve_loop_mode
from scripts.train_ar import ActivationArtifact
from scripts.train_qwen_av_reward_rl import (
    RewardNormalizer,
    build_manifest_payload,
    metric_row,
    normalized_mse_per_example,
    policy_loss_from_advantage,
    reconstruction_reward,
    resolve_checkpoint_mode,
    validation_per_example_rows,
)


def fake_artifact(num_examples: int = 2, activation_dim: int = 3) -> ActivationArtifact:
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
        manifest={},
    )


def test_normalized_mse_reward_values() -> None:
    original = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    reconstructed = torch.tensor([[1.0, 0.0], [2.0, 0.0]])

    normalized_mse = normalized_mse_per_example(original, reconstructed)
    reward = reconstruction_reward(original, reconstructed)

    assert normalized_mse[0].item() == pytest.approx(0.0)
    assert normalized_mse[1].item() == pytest.approx(1.0)
    assert torch.equal(reward.reward, -normalized_mse)
    assert set(reward.__dataclass_fields__) == {
        "reward",
        "normalized_mse",
        "raw_mse",
        "cosine_similarity",
        "reconstructed",
    }


def test_reward_normalization_modes() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])

    none = RewardNormalizer("none").normalize(rewards)
    zscore = RewardNormalizer("batch_zscore").normalize(rewards)
    ema_normalizer = RewardNormalizer("ema")
    ema_first = ema_normalizer.normalize(torch.tensor([2.0]))
    ema_second = ema_normalizer.normalize(torch.tensor([4.0]))

    assert torch.equal(none, rewards)
    assert zscore.mean().item() == pytest.approx(0.0)
    assert zscore.std(unbiased=False).item() == pytest.approx(1.0)
    assert ema_first.item() == pytest.approx(0.0)
    assert ema_second.item() > 0.0


def test_policy_loss_shape() -> None:
    logprobs = torch.tensor([-0.5, -1.0], requires_grad=True)
    advantages = torch.tensor([1.0, -0.5])

    loss = policy_loss_from_advantage(
        mean_token_logprobs=logprobs,
        advantages=advantages,
    )

    assert loss.ndim == 0
    loss.backward()
    assert logprobs.grad is not None


def test_validation_per_example_row_schema() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=2)
    generated_rows = [
        {"generated_text": "explanation 0"},
        {"generated_text": "explanation 1"},
    ]

    rows = validation_per_example_rows(
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
        "normalized_mse",
        "reward",
        "generated_text",
    }
    assert rows[0]["reward"] == pytest.approx(0.0)


def test_metric_row_schema() -> None:
    row = metric_row(
        epoch=1,
        train_metrics={
            "policy_loss": 0.1,
            "sft_loss": 0.2,
            "entropy": 0.3,
            "reward_mean": -0.1,
            "reward_std": 0.01,
            "normalized_mse_mean": 0.1,
            "raw_mse_mean": 0.2,
            "generated_text_mean_length": 12.0,
        },
        validation_metrics={
            "validation_nla_fve": 0.4,
            "validation_normalized_mse": 0.05,
            "validation_reward_mean": -0.05,
        },
        is_best=True,
    )

    assert row["epoch"] == 1
    assert row["validation_reward_mean"] == pytest.approx(-0.05)
    assert row["is_best"] is True


def test_checkpoint_mode_validation() -> None:
    assert (
        resolve_checkpoint_mode(
            SimpleNamespace(
                joint_checkpoint_dir="joint",
                qwen_av_checkpoint_dir=None,
                qwen_ar_checkpoint_dir=None,
            )
        )
        == "joint"
    )
    assert (
        resolve_checkpoint_mode(
            SimpleNamespace(
                joint_checkpoint_dir=None,
                qwen_av_checkpoint_dir="av",
                qwen_ar_checkpoint_dir="ar",
            )
        )
        == "split"
    )
    with pytest.raises(ValueError, match="either --joint_checkpoint_dir"):
        resolve_checkpoint_mode(
            SimpleNamespace(
                joint_checkpoint_dir="joint",
                qwen_av_checkpoint_dir="av",
                qwen_ar_checkpoint_dir=None,
            )
        )


def test_loop_runner_accepts_rl_av_split_mode() -> None:
    args = SimpleNamespace(
        joint_checkpoint_dir=None,
        qwen_av_checkpoint_dir=None,
        qwen_ar_checkpoint_dir="ar",
        rl_av_checkpoint_dir="rl_av",
    )

    assert resolve_loop_mode(args) == "rl_split"


def test_manifest_schema() -> None:
    artifact = fake_artifact(num_examples=2, activation_dim=3)
    args = {
        "activation_dir": "train_dir",
        "validation_activation_dir": "validation_dir",
        "output_dir": "out_dir",
        "joint_checkpoint_dir": "joint_dir",
        "qwen_av_checkpoint_dir": None,
        "qwen_ar_checkpoint_dir": None,
        "reward_normalization": "batch_zscore",
        "temperature": 0.7,
        "top_p": 0.95,
        "num_return_sequences": 1,
        "max_new_tokens": 64,
        "kl_weight": 0.01,
        "entropy_weight": 0.0,
        "length_penalty_weight": 0.0,
    }

    manifest = build_manifest_payload(
        args=args,
        train_artifact=artifact,
        validation_artifact=artifact,
        av_checkpoint_summary={"config": {"component": "qwen_av"}},
        ar_checkpoint_summary={"config": {"component": "qwen_ar"}},
        best_epoch=1,
        best_metrics={"validation_reward_mean": -0.1},
        output_files={"model": "model.pt"},
    )

    assert manifest["schema_version"] == "phase11_qwen_av_reward_rl_v1"
    assert manifest["checkpoint_mode"] == "joint"
    assert manifest["reward"] == "negative_l2_normalized_mse"
    assert manifest["sampling"]["max_new_tokens"] == 64
