"""Model components for activation reconstruction experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def prepend_activation_embedding(
    *,
    token_embeddings: torch.Tensor,
    activation_embedding: torch.Tensor,
) -> torch.Tensor:
    """Prepend one projected activation embedding to token embeddings."""
    if token_embeddings.ndim != 3:
        raise ValueError(
            "token_embeddings must have shape [batch, sequence, hidden], "
            f"got {token_embeddings.shape}"
        )
    if activation_embedding.ndim != 2:
        raise ValueError(
            "activation_embedding must have shape [batch, hidden], "
            f"got {activation_embedding.shape}"
        )
    if token_embeddings.shape[0] != activation_embedding.shape[0]:
        raise ValueError(
            "Batch mismatch between token_embeddings and activation_embedding: "
            f"{token_embeddings.shape[0]} vs {activation_embedding.shape[0]}"
        )
    if token_embeddings.shape[2] != activation_embedding.shape[1]:
        raise ValueError(
            "Hidden dimension mismatch between token_embeddings and activation_embedding: "
            f"{token_embeddings.shape[2]} vs {activation_embedding.shape[1]}"
        )
    return torch.cat([activation_embedding.unsqueeze(1), token_embeddings], dim=1)


def prepend_activation_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Prepend an attended pseudo-token to an attention mask."""
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [batch, sequence], got {attention_mask.shape}"
        )
    prefix = torch.ones(
        (attention_mask.shape[0], 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([prefix, attention_mask], dim=1)


def build_activation_lm_labels(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build LM labels while ignoring pseudo-token and padding positions."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, sequence], got {input_ids.shape}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape {attention_mask.shape} does not match "
            f"input_ids {input_ids.shape}"
        )
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, ignore_index)
    prefix = torch.full(
        (labels.shape[0], 1),
        ignore_index,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([prefix, labels], dim=1)


class ActivationVerbalizer(nn.Module):
    """Project activations into a causal LM prefix embedding and generate text."""

    def __init__(
        self,
        *,
        text_model_name_or_path: str,
        activation_dim: int,
        freeze_lm: bool = False,
        trust_remote_code: bool = False,
        language_model: nn.Module | None = None,
        lm_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        if activation_dim <= 0:
            raise ValueError(f"activation_dim must be positive, got {activation_dim}")

        self.text_model_name_or_path = text_model_name_or_path
        self.activation_dim = activation_dim
        self.freeze_lm = freeze_lm

        if language_model is None:
            language_model = self._load_language_model(
                text_model_name_or_path=text_model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
        self.language_model = language_model

        embedding_dim = lm_embedding_dim or _infer_lm_embedding_dim(self.language_model)
        if embedding_dim <= 0:
            raise ValueError(f"LM embedding dimension must be positive, got {embedding_dim}")
        self.lm_embedding_dim = embedding_dim
        self.activation_projection = nn.Linear(activation_dim, embedding_dim)

        if freeze_lm:
            freeze_module(self.language_model)

    def project_activation(self, activations: torch.Tensor) -> torch.Tensor:
        """Project activation vectors into the LM embedding space."""
        if activations.ndim != 2:
            raise ValueError(
                f"activations must have shape [batch, activation_dim], got {activations.shape}"
            )
        if activations.shape[1] != self.activation_dim:
            raise ValueError(
                f"Expected activation_dim={self.activation_dim}, got {activations.shape[1]}"
            )
        return self.activation_projection(activations.float())

    def forward(
        self,
        *,
        activations: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        token_embeddings = self.language_model.get_input_embeddings()(input_ids)
        activation_embedding = self.project_activation(activations).to(
            dtype=token_embeddings.dtype,
            device=token_embeddings.device,
        )
        inputs_embeds = prepend_activation_embedding(
            token_embeddings=token_embeddings,
            activation_embedding=activation_embedding,
        )
        extended_attention_mask = prepend_activation_attention_mask(attention_mask)
        extended_labels = None
        if labels is not None:
            extended_labels = build_activation_lm_labels(
                input_ids=labels,
                attention_mask=attention_mask,
            )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
        )

    @torch.no_grad()
    def greedy_generate(
        self,
        *,
        activations: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy generation from the activation pseudo-token."""
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        embedding_layer = self.language_model.get_input_embeddings()
        generated = torch.empty(
            (activations.shape[0], 0),
            dtype=torch.long,
            device=activations.device,
        )
        finished = torch.zeros(activations.shape[0], dtype=torch.bool, device=activations.device)
        activation_embedding = self.project_activation(activations).to(
            dtype=embedding_layer.weight.dtype,
            device=activations.device,
        )

        for _step in range(max_new_tokens):
            if generated.shape[1] > 0:
                token_embeddings = embedding_layer(generated)
                inputs_embeds = prepend_activation_embedding(
                    token_embeddings=token_embeddings,
                    activation_embedding=activation_embedding,
                )
            else:
                inputs_embeds = activation_embedding.unsqueeze(1)
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished |= next_token == eos_token_id
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return generated

    @staticmethod
    def _load_language_model(
        *,
        text_model_name_or_path: str,
        trust_remote_code: bool,
    ) -> nn.Module:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Could not import transformers. Install project dependencies with "
                "`pip install -r requirements.txt` inside the project environment."
            ) from exc
        return AutoModelForCausalLM.from_pretrained(
            text_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )


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


def _infer_lm_embedding_dim(language_model: nn.Module) -> int:
    embedding_layer = language_model.get_input_embeddings()
    embedding_dim = getattr(embedding_layer, "embedding_dim", None)
    if embedding_dim is not None:
        return int(embedding_dim)
    weight = getattr(embedding_layer, "weight", None)
    if weight is not None and weight.ndim == 2:
        return int(weight.shape[1])

    config = getattr(language_model, "config", None)
    for field_name in ("n_embd", "hidden_size", "d_model"):
        value = getattr(config, field_name, None)
        if value is not None:
            return int(value)
    raise ValueError(
        "Could not infer LM embedding dimension. Pass lm_embedding_dim explicitly."
    )


def _last_hidden_state_from_output(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise ValueError("Text model output does not contain last_hidden_state.")
