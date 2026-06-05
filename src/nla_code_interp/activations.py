"""Activation extraction utilities."""

from __future__ import annotations

import torch


def select_final_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the final non-padding token index for each sequence.

    Args:
        attention_mask: Tensor with shape ``[batch_size, seq_len]`` where non-zero
            entries mark real tokens.

    Returns:
        Tensor with shape ``[batch_size]`` containing the last non-padding index
        for each row.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [batch_size, seq_len], got {attention_mask.shape}"
        )

    batch_size, seq_len = attention_mask.shape
    if batch_size == 0 or seq_len == 0:
        raise ValueError(f"attention_mask must be non-empty, got {attention_mask.shape}")

    non_padding = attention_mask != 0
    if not torch.all(non_padding.any(dim=1)):
        bad_rows = torch.where(~non_padding.any(dim=1))[0].detach().cpu().tolist()
        raise ValueError(
            "Every sequence must contain at least one non-padding token; "
            f"bad rows: {bad_rows}"
        )

    positions = torch.arange(seq_len, device=attention_mask.device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(batch_size, seq_len)
    return positions.masked_fill(~non_padding, -1).max(dim=1).values


def select_final_non_padding_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Backward-compatible singular alias for ``select_final_non_padding_indices``."""
    return select_final_non_padding_indices(attention_mask)


def select_token_activations(
    hidden_state: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """Select one activation vector per batch row with safe batched indexing."""
    if hidden_state.ndim != 3:
        raise ValueError(
            "hidden_state must have shape [batch_size, seq_len, hidden_dim], "
            f"got {hidden_state.shape}"
        )
    if token_indices.ndim != 1:
        raise ValueError(f"token_indices must have shape [batch_size], got {token_indices.shape}")

    batch_size, seq_len, _hidden_dim = hidden_state.shape
    if token_indices.shape[0] != batch_size:
        raise ValueError(
            f"Batch mismatch: hidden_state has batch size {batch_size}, "
            f"token_indices has batch size {token_indices.shape[0]}"
        )
    if batch_size == 0 or seq_len == 0:
        raise ValueError(f"hidden_state must be non-empty, got {hidden_state.shape}")

    indices = token_indices.to(device=hidden_state.device, dtype=torch.long)
    if torch.any(indices < 0) or torch.any(indices >= seq_len):
        bad = indices[(indices < 0) | (indices >= seq_len)].detach().cpu().tolist()
        raise ValueError(f"token_indices out of range for seq_len={seq_len}: {bad}")

    rows = torch.arange(batch_size, device=hidden_state.device, dtype=torch.long)
    return hidden_state[rows, indices]


def summarize_activation(activation: torch.Tensor) -> dict[str, float]:
    """Return stable summary statistics for an activation tensor."""
    if activation.numel() == 0:
        raise ValueError("activation must be non-empty")

    values = activation.detach().float()
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]

    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "l2_norm": values.norm(p=2).item(),
    }
