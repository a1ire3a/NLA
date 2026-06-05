"""Qwen-based AR and AV model components for aligned NLA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nla_code_interp.models import (
    build_activation_lm_labels,
    count_trainable_parameters,
    prepend_activation_attention_mask,
    prepend_activation_embedding,
)


DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES

    @property
    def enabled(self) -> bool:
        return self.r > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }


def dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; expected one of {sorted(mapping)}")
    return mapping[dtype_name]


def infer_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for field_name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, field_name, None)
        if value is not None:
            return int(value)

    embedding_layer = model.get_input_embeddings()
    embedding_dim = getattr(embedding_layer, "embedding_dim", None)
    if embedding_dim is not None:
        return int(embedding_dim)
    weight = getattr(embedding_layer, "weight", None)
    if weight is not None and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError("Could not infer Qwen hidden size from model config or embeddings.")


def final_non_padding_pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Select the final non-padding token representation from a hidden-state tensor."""
    if hidden_state.ndim != 3:
        raise ValueError(f"hidden_state must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence]")
    if hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError(
            f"hidden_state shape {hidden_state.shape[:2]} does not match "
            f"attention_mask {attention_mask.shape}"
        )
    token_counts = attention_mask.long().sum(dim=1)
    if (token_counts == 0).any():
        raise ValueError("Each attention_mask row must contain at least one token.")
    indices = token_counts - 1
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, indices.to(hidden_state.device), :]


def masked_mean_pool(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token representations while ignoring padding."""
    if hidden_state.ndim != 3:
        raise ValueError(f"hidden_state must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence]")
    if hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError(
            f"hidden_state shape {hidden_state.shape[:2]} does not match "
            f"attention_mask {attention_mask.shape}"
        )
    mask = attention_mask.to(device=hidden_state.device, dtype=hidden_state.dtype)
    lengths = mask.sum(dim=1, keepdim=True)
    if (lengths == 0).any():
        raise ValueError("Each attention_mask row must contain at least one token.")
    return (hidden_state * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1.0)


def output_hidden_state(outputs: Any) -> torch.Tensor:
    """Extract the last hidden-state tensor from a HF-style output object."""
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is not None:
        if len(hidden_states) == 0:
            raise ValueError("Model output hidden_states is empty.")
        return hidden_states[-1]

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None and isinstance(outputs, dict):
        last_hidden_state = outputs.get("last_hidden_state")
    if last_hidden_state is not None:
        return last_hidden_state

    if isinstance(outputs, tuple) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor) and first.ndim == 3:
            return first
    raise ValueError("Model output does not contain hidden states.")


def apply_lora(
    model: nn.Module,
    *,
    lora_settings: LoraSettings,
) -> nn.Module:
    """Apply PEFT LoRA adapters to a Qwen model, or freeze base model when disabled."""
    if not lora_settings.enabled:
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "Could not import peft. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    config = LoraConfig(
        r=lora_settings.r,
        lora_alpha=lora_settings.alpha,
        lora_dropout=lora_settings.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(lora_settings.target_modules),
    )
    return get_peft_model(model, config)


def load_qwen_causal_lm(
    *,
    model_name_or_path: str,
    dtype: torch.dtype,
    local_files_only: bool = False,
) -> nn.Module:
    """Load Qwen with the dtype-first Transformers API and torch_dtype fallback."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformers. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    kwargs = {
        "local_files_only": local_files_only,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            **kwargs,
        )


class QwenActivationReconstructor(nn.Module):
    """Qwen text encoder/causal LM pooled into an activation reconstruction head."""

    def __init__(
        self,
        *,
        qwen_model: nn.Module,
        activation_dim: int,
        pooling: str = "final",
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if activation_dim <= 0:
            raise ValueError(f"activation_dim must be positive, got {activation_dim}")
        if pooling not in {"final", "mean"}:
            raise ValueError(f"Unsupported pooling {pooling!r}; expected 'final' or 'mean'.")
        self.qwen_model = qwen_model
        self.activation_dim = activation_dim
        self.pooling = pooling
        self.hidden_size = hidden_size or infer_hidden_size(qwen_model)
        self.projection = nn.Linear(self.hidden_size, activation_dim)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_state = output_hidden_state(outputs)
        if self.pooling == "final":
            pooled = final_non_padding_pool(hidden_state, attention_mask)
        else:
            pooled = masked_mean_pool(hidden_state, attention_mask)
        return self.projection(pooled.float())


class QwenActivationVerbalizer(nn.Module):
    """Qwen causal LM with one projected activation pseudo-token prefix."""

    def __init__(
        self,
        *,
        qwen_model: nn.Module,
        activation_dim: int,
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        if activation_dim <= 0:
            raise ValueError(f"activation_dim must be positive, got {activation_dim}")
        self.qwen_model = qwen_model
        self.activation_dim = activation_dim
        self.embedding_dim = embedding_dim or infer_hidden_size(qwen_model)
        self.activation_projection = nn.Linear(activation_dim, self.embedding_dim)

    def project_activation(self, activations: torch.Tensor) -> torch.Tensor:
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
        embedding_layer = self.qwen_model.get_input_embeddings()
        token_embeddings = embedding_layer(input_ids)
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
        return self.qwen_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            use_cache=False,
        )

    @torch.no_grad()
    def greedy_generate(
        self,
        *,
        activations: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        embedding_layer = self.qwen_model.get_input_embeddings()
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
            outputs = self.qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
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


def qwen_checkpoint_metadata(
    *,
    component: str,
    model_name_or_path: str,
    activation_dim: int,
    dtype: str,
    lora_settings: LoraSettings,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if component not in {"qwen_ar", "qwen_av"}:
        raise ValueError(f"Unsupported Qwen component {component!r}")
    config = {
        "component": component,
        "model_name_or_path": model_name_or_path,
        "activation_dim": activation_dim,
        "dtype": dtype,
        "lora": lora_settings.as_dict(),
    }
    if extra_config:
        config.update(extra_config)
    return config


def qwen_trainable_parameter_summary(module: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = count_trainable_parameters(module)
    return {"total": total, "trainable": trainable}
