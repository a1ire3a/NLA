"""Qwen-based AR and AV model components for aligned NLA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class QwenTargetTransform:
    """Target-space transform restored from a Qwen AR checkpoint."""

    name: str
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    eps: float = 1e-6

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        targets = validate_activation_matrix(targets, "targets")
        if self.name == "raw":
            return targets.clone()
        if self.mean is None:
            raise ValueError(f"Target transform {self.name!r} is missing mean.")
        mean = self.mean.to(device=targets.device, dtype=targets.dtype)
        centered = targets - mean
        if self.name == "center":
            return centered
        if self.std is None:
            raise ValueError("standardize target transform is missing std.")
        std = self.std.to(device=targets.device, dtype=targets.dtype)
        return centered / std

    def inverse_transform(self, predictions: torch.Tensor) -> torch.Tensor:
        predictions = validate_activation_matrix(predictions, "predictions")
        if self.name == "raw":
            return predictions.clone()
        if self.mean is None:
            raise ValueError(f"Target transform {self.name!r} is missing mean.")
        mean = self.mean.to(device=predictions.device, dtype=predictions.dtype)
        if self.name == "center":
            return predictions + mean
        if self.std is None:
            raise ValueError("standardize target transform is missing std.")
        std = self.std.to(device=predictions.device, dtype=predictions.dtype)
        return predictions * std + mean

    def state_dict_for_checkpoint(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "eps": self.eps,
            "mean": self.mean.detach().cpu() if self.mean is not None else None,
            "std": self.std.detach().cpu() if self.std is not None else None,
        }

    def state_dict_for_manifest(self) -> dict[str, Any]:
        state: dict[str, Any] = {"name": self.name, "eps": self.eps}
        if self.mean is not None:
            mean = self.mean.detach().cpu().float()
            state.update({"mean_shape": list(mean.shape), "mean_norm": mean.norm().item()})
        if self.std is not None:
            std = self.std.detach().cpu().float()
            state.update(
                {
                    "std_shape": list(std.shape),
                    "std_mean": std.mean().item(),
                    "std_min": std.min().item(),
                    "std_max": std.max().item(),
                }
            )
        return state


@dataclass(frozen=True)
class QwenAVCheckpointBundle:
    model: "QwenActivationVerbalizer"
    tokenizer: Any
    config: dict[str, Any]
    checkpoint: dict[str, Any]


@dataclass(frozen=True)
class QwenARCheckpointBundle:
    model: "QwenActivationReconstructor"
    tokenizer: Any
    target_transform: QwenTargetTransform
    config: dict[str, Any]
    checkpoint: dict[str, Any]


def dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; expected one of {sorted(mapping)}")
    return mapping[dtype_name]


def validate_activation_matrix(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [num_examples, activation_dim]")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {tuple(tensor.shape)}")
    return tensor.detach().float()


def tensor_from_transform_state(value: Any, *, name: str) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    elif isinstance(value, list):
        tensor = torch.tensor(value, dtype=torch.float32)
    else:
        raise TypeError(f"target_transform_state[{name!r}] must be a tensor or list.")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError(
            f"target_transform_state[{name!r}] must have shape [1, activation_dim] "
            f"or [activation_dim], got {tuple(tensor.shape)}"
        )
    return tensor


def target_transform_from_checkpoint_state(state: dict[str, Any]) -> QwenTargetTransform:
    if not isinstance(state, dict):
        raise TypeError(f"target_transform_state must be a dict, got {type(state)}")
    name = state.get("name")
    if name not in {"raw", "center", "standardize"}:
        raise ValueError(f"Unsupported target transform in checkpoint: {name!r}")
    mean = tensor_from_transform_state(state.get("mean"), name="mean")
    std = tensor_from_transform_state(state.get("std"), name="std")
    eps = float(state.get("eps", 1e-6))
    if name in {"center", "standardize"} and mean is None:
        raise ValueError(f"{name} target transform checkpoint is missing mean.")
    if name == "standardize" and std is None:
        raise ValueError("standardize target transform checkpoint is missing std.")
    return QwenTargetTransform(name=name, mean=mean, std=std, eps=eps)


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


def load_checkpoint_payload(checkpoint_dir: Path, *, device: torch.device | str) -> dict[str, Any]:
    checkpoint_path = checkpoint_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Qwen checkpoint file: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict in {checkpoint_path}, got {type(checkpoint)}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{checkpoint_path} is missing config.")
    output_files = checkpoint.get("output_files")
    if not isinstance(output_files, dict):
        raise ValueError(f"{checkpoint_path} is missing output_files.")
    return checkpoint


def load_qwen_tokenizer_from_checkpoint(
    *,
    checkpoint_dir: Path,
    config: dict[str, Any],
    output_files: dict[str, Any],
):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformers. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    tokenizer_path = checkpoint_dir / output_files.get("tokenizer", "tokenizer")
    tokenizer_source = (
        str(tokenizer_path) if tokenizer_path.exists() else config["model_name_or_path"]
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad token and no eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qwen_model_from_checkpoint(
    *,
    checkpoint_dir: Path,
    config: dict[str, Any],
    output_files: dict[str, Any],
    adapter_trainable: bool = False,
) -> nn.Module:
    qwen_model = load_qwen_causal_lm(
        model_name_or_path=config["model_name_or_path"],
        dtype=dtype_from_name(config.get("dtype", "bfloat16")),
    )
    lora_config = config.get("lora", {})
    if lora_config.get("enabled", False):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Could not import peft. Install project dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        adapter_dir = checkpoint_dir / output_files.get("qwen_adapter", "qwen_adapter")
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Missing Qwen adapter directory: {adapter_dir}")
        return PeftModel.from_pretrained(
            qwen_model,
            adapter_dir,
            is_trainable=adapter_trainable,
        )

    if "qwen_model_state" in output_files:
        state_path = checkpoint_dir / output_files["qwen_model_state"]
        qwen_model.load_state_dict(torch.load(state_path, map_location="cpu"))
    for parameter in qwen_model.parameters():
        parameter.requires_grad = False
    return qwen_model


def load_qwen_av_checkpoint(
    *,
    checkpoint_dir: Path,
    device: torch.device,
    adapter_trainable: bool = False,
) -> QwenAVCheckpointBundle:
    checkpoint = load_checkpoint_payload(checkpoint_dir, device=device)
    config = checkpoint["config"]
    if config.get("component") != "qwen_av":
        raise ValueError(f"Expected qwen_av checkpoint, got {config.get('component')!r}")
    output_files = checkpoint["output_files"]
    tokenizer = load_qwen_tokenizer_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        config=config,
        output_files=output_files,
    )
    qwen_model = load_qwen_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        config=config,
        output_files=output_files,
        adapter_trainable=adapter_trainable,
    )
    model = QwenActivationVerbalizer(
        qwen_model=qwen_model,
        activation_dim=int(config["activation_dim"]),
    )
    projection_path = checkpoint_dir / output_files["activation_projection"]
    model.activation_projection.load_state_dict(
        torch.load(projection_path, map_location="cpu")
    )
    model = model.to(device)
    model.eval()
    return QwenAVCheckpointBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint=checkpoint,
    )


def load_qwen_ar_checkpoint(
    *,
    checkpoint_dir: Path,
    device: torch.device,
    adapter_trainable: bool = False,
) -> QwenARCheckpointBundle:
    checkpoint = load_checkpoint_payload(checkpoint_dir, device=device)
    config = checkpoint["config"]
    if config.get("component") != "qwen_ar":
        raise ValueError(f"Expected qwen_ar checkpoint, got {config.get('component')!r}")
    output_files = checkpoint["output_files"]
    transform_state = checkpoint.get("target_transform_state")
    if not isinstance(transform_state, dict):
        raise ValueError(f"{checkpoint_dir / 'model.pt'} is missing target_transform_state.")
    tokenizer = load_qwen_tokenizer_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        config=config,
        output_files=output_files,
    )
    qwen_model = load_qwen_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        config=config,
        output_files=output_files,
        adapter_trainable=adapter_trainable,
    )
    model = QwenActivationReconstructor(
        qwen_model=qwen_model,
        activation_dim=int(config["activation_dim"]),
        pooling=config.get("pooling", "final"),
    )
    projection_path = checkpoint_dir / output_files["projection_head"]
    model.projection.load_state_dict(torch.load(projection_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    return QwenARCheckpointBundle(
        model=model,
        tokenizer=tokenizer,
        target_transform=target_transform_from_checkpoint_state(transform_state),
        config=config,
        checkpoint=checkpoint,
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
