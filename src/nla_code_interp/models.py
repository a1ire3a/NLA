"""Model components for activation reconstruction experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ActivationVerbalizerPlaceholder:
    """Placeholder for activation-to-text model design."""

    pass


def mean_pool_last_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token representations while ignoring padded tokens."""
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state must have shape [batch, sequence, hidden], "
            f"got {last_hidden_state.shape}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [batch, sequence], got {attention_mask.shape}"
        )
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError(
            "last_hidden_state and attention_mask batch/sequence dimensions must match: "
            f"{last_hidden_state.shape[:2]} vs {attention_mask.shape}"
        )
    if (attention_mask.sum(dim=1) == 0).any():
        raise ValueError("Each row must contain at least one non-padding token.")

    mask = attention_mask.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype)
    masked_hidden = last_hidden_state * mask.unsqueeze(-1)
    lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return masked_hidden.sum(dim=1) / lengths


def freeze_module(module: nn.Module) -> None:
    """Disable gradients for every parameter in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = False


def count_trainable_parameters(module: nn.Module) -> int:
    """Count parameters that will receive optimizer updates."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class TextActivationReconstructor(nn.Module):
    """Encode text and project it into a target activation vector space."""

    def __init__(
        self,
        *,
        text_model_name_or_path: str,
        activation_dim: int,
        pooling: str = "mean",
        projection_hidden_dim: int | None = None,
        dropout: float = 0.0,
        freeze_text_model: bool = True,
        trust_remote_code: bool = False,
        text_model: nn.Module | None = None,
        text_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if activation_dim <= 0:
            raise ValueError(f"activation_dim must be positive, got {activation_dim}")
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling={pooling!r}; expected 'mean' or 'cls'.")
        if projection_hidden_dim is not None and projection_hidden_dim <= 0:
            raise ValueError(
                "projection_hidden_dim must be positive when set, "
                f"got {projection_hidden_dim}"
            )
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.text_model_name_or_path = text_model_name_or_path
        self.activation_dim = activation_dim
        self.pooling = pooling
        self.projection_hidden_dim = projection_hidden_dim
        self.dropout = dropout
        self.freeze_text_model = freeze_text_model

        if text_model is None:
            text_model = self._load_text_model(
                text_model_name_or_path=text_model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
        self.text_model = text_model

        hidden_dim = text_hidden_dim or _infer_text_hidden_dim(self.text_model)
        if hidden_dim <= 0:
            raise ValueError(f"text hidden dimension must be positive, got {hidden_dim}")
        self.text_hidden_dim = hidden_dim
        self.projection = _build_projection_head(
            text_hidden_dim=hidden_dim,
            activation_dim=activation_dim,
            projection_hidden_dim=projection_hidden_dim,
            dropout=dropout,
        )

        if freeze_text_model:
            freeze_module(self.text_model)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = _last_hidden_state_from_output(outputs)
        if self.pooling == "mean":
            pooled = mean_pool_last_hidden_state(last_hidden_state, attention_mask)
        else:
            pooled = last_hidden_state[:, 0, :]
        return self.projection(pooled)

    @staticmethod
    def _load_text_model(
        *,
        text_model_name_or_path: str,
        trust_remote_code: bool,
    ) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Could not import transformers. Install project dependencies with "
                "`pip install -r requirements.txt` inside the project environment."
            ) from exc
        return AutoModel.from_pretrained(
            text_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )


def _build_projection_head(
    *,
    text_hidden_dim: int,
    activation_dim: int,
    projection_hidden_dim: int | None,
    dropout: float,
) -> nn.Module:
    if projection_hidden_dim is None:
        return nn.Linear(text_hidden_dim, activation_dim)
    return nn.Sequential(
        nn.Linear(text_hidden_dim, projection_hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(projection_hidden_dim, activation_dim),
    )


def _infer_text_hidden_dim(text_model: nn.Module) -> int:
    config = getattr(text_model, "config", None)
    for field_name in ("hidden_size", "dim", "d_model"):
        value = getattr(config, field_name, None)
        if value is not None:
            return int(value)
    raise ValueError(
        "Could not infer text model hidden size from config. Pass text_hidden_dim explicitly."
    )


def _last_hidden_state_from_output(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise ValueError("Text model output does not contain last_hidden_state.")
