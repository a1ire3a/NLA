from __future__ import annotations

import pytest
import torch

from nla_code_interp.metrics import (
    baseline_mean_reconstruction,
    baseline_shuffled_reconstruction,
    baseline_zero_reconstruction,
    cosine_similarity_summary,
    fraction_variance_explained,
    per_example_cosine_similarity,
    per_example_l2_error,
    per_example_squared_error,
    summarize_reconstruction,
)


def test_fraction_variance_explained_perfect_reconstruction() -> None:
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = original.clone()

    score = fraction_variance_explained(original, reconstructed)

    assert torch.isclose(score, torch.tensor(1.0))


def test_fraction_variance_explained_mean_baseline_is_zero() -> None:
    original = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    reconstructed = baseline_mean_reconstruction(original, original)

    score = fraction_variance_explained(original, reconstructed)

    assert score.item() == pytest.approx(0.0, abs=1e-6)


def test_fraction_variance_explained_bad_reconstruction_negative() -> None:
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = torch.tensor([[10.0, 10.0], [10.0, 10.0]])

    score = fraction_variance_explained(original, reconstructed)

    assert score.item() < 0.0


def test_per_example_error_shapes() -> None:
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = torch.zeros_like(original)

    assert per_example_squared_error(original, reconstructed).shape == (2,)
    assert per_example_l2_error(original, reconstructed).shape == (2,)
    assert per_example_cosine_similarity(original, reconstructed).shape == (2,)


def test_baseline_shapes_and_deterministic_shuffle() -> None:
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    zero = baseline_zero_reconstruction(target)
    shuffled_1 = baseline_shuffled_reconstruction(target, seed=42)
    shuffled_2 = baseline_shuffled_reconstruction(target, seed=42)

    assert zero.shape == target.shape
    assert shuffled_1.shape == target.shape
    assert torch.equal(shuffled_1, shuffled_2)


def test_cosine_summary_returns_expected_keys() -> None:
    original = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reconstructed = original.clone()

    summary = cosine_similarity_summary(original, reconstructed)

    assert set(summary) == {"cosine_mean", "cosine_std", "cosine_min", "cosine_max"}
    assert summary["cosine_mean"] == pytest.approx(1.0)


def test_summarize_reconstruction_keys() -> None:
    original = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reconstructed = original.clone()

    summary = summarize_reconstruction(original, reconstructed)

    assert set(summary) == {
        "fve",
        "mse",
        "rmse",
        "mean_l2_error",
        "cosine_mean",
        "cosine_std",
        "cosine_min",
        "cosine_max",
    }
    assert summary["fve"] == pytest.approx(1.0)
