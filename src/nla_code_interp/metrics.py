"""Evaluation metrics for activation reconstruction."""

from __future__ import annotations

import torch


def fraction_variance_explained(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute Fraction of Variance Explained (FVE).

    This placeholder implementation assumes tensors have shape `[num_examples, activation_dim]`.
    It will be reviewed and tested before final experiments.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape=} vs {reconstructed.shape=}")

    mean_original = original.mean(dim=0, keepdim=True)
    reconstruction_error = torch.sum((original - reconstructed) ** 2)
    baseline_error = torch.sum((original - mean_original) ** 2)

    if baseline_error == 0:
        raise ValueError("Cannot compute FVE when baseline variance is zero.")

    return 1.0 - reconstruction_error / baseline_error
