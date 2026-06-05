"""Evaluation metrics for activation reconstruction."""

from __future__ import annotations

import math

import torch


def fraction_variance_explained(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute Fraction of Variance Explained (FVE)."""
    original_f, reconstructed_f = _validate_pair(original, reconstructed)
    mean_original = original_f.mean(dim=0, keepdim=True)
    reconstruction_error = torch.sum((original_f - reconstructed_f) ** 2)
    baseline_error = torch.sum((original_f - mean_original) ** 2)
    return 1.0 - reconstruction_error / (baseline_error + eps)


def per_example_squared_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    """Return summed squared error per example."""
    original_f, reconstructed_f = _validate_pair(original, reconstructed)
    return torch.sum((original_f - reconstructed_f) ** 2, dim=1)


def per_example_l2_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    """Return L2 reconstruction error per example."""
    return torch.sqrt(per_example_squared_error(original, reconstructed))


def per_example_cosine_similarity(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return cosine similarity per example with safe normalization."""
    original_f, reconstructed_f = _validate_pair(original, reconstructed)
    numerator = torch.sum(original_f * reconstructed_f, dim=1)
    original_norm = original_f.norm(p=2, dim=1)
    reconstructed_norm = reconstructed_f.norm(p=2, dim=1)
    denominator = torch.clamp(original_norm * reconstructed_norm, min=eps)
    return numerator / denominator


def cosine_similarity_summary(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float]:
    """Summarize per-example cosine similarities."""
    similarities = per_example_cosine_similarity(original, reconstructed)
    return {
        "cosine_mean": similarities.mean().item(),
        "cosine_std": similarities.std(unbiased=False).item(),
        "cosine_min": similarities.min().item(),
        "cosine_max": similarities.max().item(),
    }


def baseline_mean_reconstruction(
    train_or_reference: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Repeat the reference mean activation to match target rows."""
    reference = _validate_activation_matrix(train_or_reference, "train_or_reference")
    target_f = _validate_activation_matrix(target, "target")
    if reference.shape[1] != target_f.shape[1]:
        raise ValueError(
            f"Activation dimension mismatch: reference dim {reference.shape[1]} vs "
            f"target dim {target_f.shape[1]}"
        )
    mean_vector = reference.mean(dim=0, keepdim=True)
    return mean_vector.repeat(target_f.shape[0], 1)


def baseline_zero_reconstruction(target: torch.Tensor) -> torch.Tensor:
    """Return a zero-vector reconstruction baseline."""
    target_f = _validate_activation_matrix(target, "target")
    return torch.zeros_like(target_f)


def baseline_shuffled_reconstruction(target: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a row-shuffled reconstruction baseline."""
    target_f = _validate_activation_matrix(target, "target")
    if target_f.shape[0] == 1:
        return target_f.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(target_f.shape[0], generator=generator)
    return target_f[permutation].clone()


def summarize_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float]:
    """Summarize a reconstruction with aggregate scalar metrics."""
    original_f, reconstructed_f = _validate_pair(original, reconstructed)
    squared_errors = per_example_squared_error(original_f, reconstructed_f)
    mse = squared_errors.mean().item() / original_f.shape[1]
    cosine = cosine_similarity_summary(original_f, reconstructed_f)
    return {
        "fve": fraction_variance_explained(original_f, reconstructed_f).item(),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mean_l2_error": torch.sqrt(squared_errors).mean().item(),
        **cosine,
    }


def _validate_pair(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_f = _validate_activation_matrix(original, "original")
    reconstructed_f = _validate_activation_matrix(reconstructed, "reconstructed")
    if original_f.shape != reconstructed_f.shape:
        raise ValueError(
            f"Shape mismatch: original={original_f.shape} vs "
            f"reconstructed={reconstructed_f.shape}"
        )
    return original_f, reconstructed_f


def _validate_activation_matrix(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must have shape [num_examples, hidden_dim], got {tensor.shape}"
        )
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {tensor.shape}")
    return tensor.detach().float()
