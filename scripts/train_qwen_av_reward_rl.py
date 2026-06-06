"""Reward-driven Qwen AV fine-tuning for the NLA loop."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.data import write_jsonl  # noqa: E402
from nla_code_interp.metrics import (  # noqa: E402
    baseline_mean_reconstruction,
    baseline_shuffled_reconstruction,
    baseline_zero_reconstruction,
    per_example_cosine_similarity,
    per_example_l2_error,
    per_example_squared_error,
    summarize_reconstruction,
)
from nla_code_interp.models import (  # noqa: E402
    prepend_activation_embedding,
)
from nla_code_interp.qwen_models import (  # noqa: E402
    LoraSettings,
    QwenARCheckpointBundle,
    QwenAVCheckpointBundle,
    QwenJointCheckpointBundle,
    load_qwen_ar_checkpoint,
    load_qwen_av_checkpoint,
    load_qwen_joint_checkpoint,
    qwen_checkpoint_metadata,
    qwen_trainable_parameter_summary,
)
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.train_ar import (  # noqa: E402
    ActivationArtifact,
    ensure_tokenizer_padding,
    import_tqdm,
    load_activation_artifact,
    resolve_device,
)
from scripts.train_qwen_av import (  # noqa: E402
    append_eos,
    build_examples as build_qwen_av_examples,
    generate_rows as generate_qwen_av_rows,
    parse_fallback_fields,
)


SCHEMA_VERSION = "phase11_qwen_av_reward_rl_v1"
REWARD_NORMALIZATION_MODES = ("none", "batch_zscore", "ema")
TARGET_TEXT_FIELDS = ("reference_description", "prompt", "code")
METRIC_COLUMNS = (
    "epoch",
    "policy_loss",
    "sft_loss",
    "entropy",
    "reward_mean",
    "reward_std",
    "normalized_mse_mean",
    "raw_mse_mean",
    "generated_text_mean_length",
    "validation_nla_fve",
    "validation_normalized_mse",
    "validation_reward_mean",
    "is_best",
)


@dataclass(frozen=True)
class RLExample:
    activation: torch.Tensor
    target_text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SampledTextBatch:
    generated_ids: torch.Tensor
    token_mask: torch.Tensor
    generated_texts: list[str]


@dataclass(frozen=True)
class RewardBatch:
    reward: torch.Tensor
    normalized_mse: torch.Tensor
    raw_mse: torch.Tensor
    cosine_similarity: torch.Tensor
    reconstructed: torch.Tensor


class RLDataset(Dataset):
    def __init__(self, examples: list[RLExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RLExample:
        return self.examples[index]


class RewardNormalizer:
    def __init__(self, mode: str, *, ema_decay: float = 0.9, eps: float = 1e-6) -> None:
        if mode not in REWARD_NORMALIZATION_MODES:
            raise ValueError(f"Unsupported reward normalization mode: {mode!r}")
        self.mode = mode
        self.ema_decay = ema_decay
        self.eps = eps
        self.ema_baseline: float | None = None

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.ndim != 1:
            raise ValueError(f"rewards must have shape [batch], got {tuple(rewards.shape)}")
        if self.mode == "none":
            return rewards
        if self.mode == "batch_zscore":
            if rewards.numel() <= 1:
                return rewards
            std = rewards.std(unbiased=False)
            if std.item() < self.eps:
                return rewards - rewards.mean()
            return (rewards - rewards.mean()) / std.clamp_min(self.eps)
        batch_mean = rewards.detach().mean().item()
        if self.ema_baseline is None:
            self.ema_baseline = batch_mean
        else:
            self.ema_baseline = (
                self.ema_decay * self.ema_baseline + (1.0 - self.ema_decay) * batch_mean
            )
        return rewards - float(self.ema_baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen AV from AR reconstruction reward.")
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--validation_activation_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--joint_checkpoint_dir", default=None)
    parser.add_argument("--qwen_av_checkpoint_dir", default=None)
    parser.add_argument("--qwen_ar_checkpoint_dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate_av", type=float, default=5e-5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_ar_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--reward_normalization",
        choices=REWARD_NORMALIZATION_MODES,
        default="batch_zscore",
    )
    parser.add_argument("--kl_weight", type=float, default=0.01)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--length_penalty_weight", type=float, default=0.0)
    parser.add_argument(
        "--target_text_field",
        choices=TARGET_TEXT_FIELDS,
        default="reference_description",
    )
    parser.add_argument("--fallback_text_fields", default="prompt,code")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_validation", type=int, default=None)
    parser.add_argument("--eval_every_epoch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    mode = resolve_checkpoint_mode(args)
    if mode not in {"joint", "split"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")
    for name in (
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_new_tokens",
        "max_ar_length",
        "num_return_sequences",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.learning_rate_av <= 0.0:
        raise ValueError(f"learning_rate_av must be positive, got {args.learning_rate_av}")
    if args.temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {args.temperature}")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {args.top_p}")
    for name in ("kl_weight", "entropy_weight", "length_penalty_weight"):
        value = getattr(args, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    for name in ("limit_train", "limit_validation"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when set, got {value}")


def resolve_checkpoint_mode(args: argparse.Namespace) -> str:
    has_joint = bool(args.joint_checkpoint_dir)
    has_av = bool(args.qwen_av_checkpoint_dir)
    has_ar = bool(args.qwen_ar_checkpoint_dir)
    if has_joint and (has_av or has_ar):
        raise ValueError(
            "Use either --joint_checkpoint_dir or split --qwen_av_checkpoint_dir/"
            "--qwen_ar_checkpoint_dir, not both."
        )
    if has_joint:
        return "joint"
    if has_av and has_ar:
        return "split"
    raise ValueError(
        "Provide either --joint_checkpoint_dir or both --qwen_av_checkpoint_dir "
        "and --qwen_ar_checkpoint_dir."
    )


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is non-empty: {output_dir}. "
                "Pass --overwrite to replace it intentionally."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def subset_artifact(artifact: ActivationArtifact, limit: int | None) -> ActivationArtifact:
    if limit is None:
        return artifact
    if limit > artifact.num_examples:
        raise ValueError(
            f"limit={limit} exceeds available examples {artifact.num_examples} "
            f"in {artifact.activation_dir}"
        )
    return ActivationArtifact(
        activation_dir=artifact.activation_dir,
        activations=artifact.activations[:limit].clone(),
        metadata_rows=list(artifact.metadata_rows[:limit]),
        manifest=artifact.manifest,
    )


def load_phase11_checkpoints(
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[QwenAVCheckpointBundle, QwenARCheckpointBundle, QwenJointCheckpointBundle | None]:
    if resolve_checkpoint_mode(args) == "joint":
        joint_bundle = load_qwen_joint_checkpoint(
            checkpoint_dir=Path(args.joint_checkpoint_dir),
            device=device,
            av_adapter_trainable=True,
            ar_adapter_trainable=False,
        )
        return joint_bundle.av_bundle, joint_bundle.ar_bundle, joint_bundle
    return (
        load_qwen_av_checkpoint(
            checkpoint_dir=Path(args.qwen_av_checkpoint_dir),
            device=device,
            adapter_trainable=True,
        ),
        load_qwen_ar_checkpoint(
            checkpoint_dir=Path(args.qwen_ar_checkpoint_dir),
            device=device,
            adapter_trainable=False,
        ),
        None,
    )


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


def build_rl_examples(
    *,
    artifact: ActivationArtifact,
    target_text_field: str,
    fallback_text_fields: list[str],
) -> list[RLExample]:
    av_examples = build_qwen_av_examples(
        artifact=artifact,
        target_text_field=target_text_field,
        fallback_text_fields=fallback_text_fields,
    )
    examples = []
    for index, av_example in enumerate(av_examples):
        examples.append(
            RLExample(
                activation=artifact.activations[index],
                target_text=av_example.target_text,
                metadata=dict(av_example.metadata),
            )
        )
    return examples


def make_collate_fn(tokenizer, max_target_length: int):
    def collate(batch: list[RLExample]) -> dict[str, Any]:
        tokenized = tokenizer(
            [append_eos(item.target_text, tokenizer.eos_token) for item in batch],
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(tokenized["input_ids"])
        return {
            "activations": torch.stack([item.activation for item in batch], dim=0),
            "target_input_ids": tokenized["input_ids"],
            "target_attention_mask": attention_mask,
            "target_texts": [item.target_text for item in batch],
            "metadata": [item.metadata for item in batch],
        }

    return collate


def l2_normalize_rows(tensor: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return tensor.float() / tensor.float().norm(p=2, dim=1, keepdim=True).clamp_min(eps)


def normalized_mse_per_example(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original={tuple(original.shape)} vs "
            f"reconstructed={tuple(reconstructed.shape)}"
        )
    normalized_original = l2_normalize_rows(original)
    normalized_reconstructed = l2_normalize_rows(reconstructed)
    return ((normalized_original - normalized_reconstructed) ** 2).mean(dim=1)


def reconstruction_reward(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> RewardBatch:
    normalized_mse = normalized_mse_per_example(original, reconstructed)
    raw_mse = ((original.float() - reconstructed.float()) ** 2).mean(dim=1)
    cosine = per_example_cosine_similarity(original, reconstructed)
    return RewardBatch(
        reward=-normalized_mse,
        normalized_mse=normalized_mse,
        raw_mse=raw_mse,
        cosine_similarity=cosine,
        reconstructed=reconstructed.detach().cpu().float(),
    )


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove_sorted = cumulative > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(logits, dtype=torch.bool)
    remove.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


@torch.no_grad()
def sample_av_text(
    *,
    model,
    tokenizer,
    activations: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> SampledTextBatch:
    model.eval()
    embedding_layer = model.qwen_model.get_input_embeddings()
    generated = torch.empty(
        (activations.shape[0], 0),
        dtype=torch.long,
        device=activations.device,
    )
    token_masks = []
    finished = torch.zeros(activations.shape[0], dtype=torch.bool, device=activations.device)
    activation_embedding = model.project_activation(activations).to(
        dtype=embedding_layer.weight.dtype,
        device=activations.device,
    )
    eos_token_id = tokenizer.eos_token_id
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
        outputs = model.qwen_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, -1, :].float() / temperature
        logits = top_p_filter(logits, top_p)
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(1)
        active = ~finished
        if eos_token_id is not None:
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished |= active & (next_token == eos_token_id)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        token_masks.append(active.long())
        if eos_token_id is not None and bool(finished.all()):
            break
    if not token_masks:
        raise ValueError("Sampling produced no generated tokens.")
    token_mask = torch.stack(token_masks, dim=1)
    texts = [
        text.strip() or "."
        for text in tokenizer.batch_decode(generated.detach().cpu(), skip_special_tokens=True)
    ]
    return SampledTextBatch(
        generated_ids=generated,
        token_mask=token_mask,
        generated_texts=texts,
    )


def av_token_logprobs_and_entropy(
    *,
    model,
    activations: torch.Tensor,
    generated_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding_layer = model.qwen_model.get_input_embeddings()
    token_embeddings = embedding_layer(generated_ids)
    activation_embedding = model.project_activation(activations).to(
        dtype=token_embeddings.dtype,
        device=token_embeddings.device,
    )
    inputs_embeds = prepend_activation_embedding(
        token_embeddings=token_embeddings,
        activation_embedding=activation_embedding,
    )
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    outputs = model.qwen_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    token_logits = outputs.logits[:, : generated_ids.shape[1], :].float()
    log_probs = torch.log_softmax(token_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
    probs = torch.softmax(token_logits, dim=-1)
    token_entropy = -(probs * log_probs).sum(dim=-1)
    mask = token_mask.to(device=token_logprobs.device, dtype=token_logprobs.dtype)
    lengths = mask.sum(dim=1).clamp_min(1.0)
    return (
        (token_logprobs * mask).sum(dim=1) / lengths,
        (token_entropy * mask).sum(dim=1) / lengths,
    )


def sft_loss_for_targets(
    *,
    model,
    activations: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        activations=activations,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    if outputs.loss is None:
        raise ValueError("AV model did not return SFT loss.")
    return outputs.loss


@torch.no_grad()
def reconstruct_texts_with_ar(
    *,
    ar_bundle: QwenARCheckpointBundle,
    texts: list[str],
    device: torch.device,
    max_ar_length: int,
) -> torch.Tensor:
    tokenized = ar_bundle.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_ar_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    transformed = ar_bundle.model(input_ids=input_ids, attention_mask=attention_mask)
    return ar_bundle.target_transform.inverse_transform(transformed.detach().cpu().float())


def policy_loss_from_advantage(
    *,
    mean_token_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    if mean_token_logprobs.shape != advantages.shape:
        raise ValueError(
            f"Shape mismatch: logprobs={tuple(mean_token_logprobs.shape)} vs "
            f"advantages={tuple(advantages.shape)}"
        )
    return -(advantages.detach() * mean_token_logprobs).mean()


def repeat_for_return_sequences(tensor: torch.Tensor, num_return_sequences: int) -> torch.Tensor:
    if num_return_sequences == 1:
        return tensor
    return tensor.repeat_interleave(num_return_sequences, dim=0)


def expand_metadata(
    metadata_rows: list[dict[str, Any]],
    num_return_sequences: int,
) -> list[dict[str, Any]]:
    if num_return_sequences == 1:
        return [dict(row) for row in metadata_rows]
    expanded = []
    for row in metadata_rows:
        for sequence_index in range(num_return_sequences):
            copy = dict(row)
            copy["return_sequence_index"] = sequence_index
            expanded.append(copy)
    return expanded


def train_one_epoch(
    *,
    av_bundle: QwenAVCheckpointBundle,
    ar_bundle: QwenARCheckpointBundle,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    reward_normalizer: RewardNormalizer,
    args: argparse.Namespace,
    device: torch.device,
    tqdm,
) -> dict[str, float]:
    av_bundle.model.train()
    ar_bundle.model.eval()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "policy_loss": 0.0,
        "sft_loss": 0.0,
        "entropy": 0.0,
        "reward_mean": 0.0,
        "reward_sq_sum": 0.0,
        "normalized_mse_mean": 0.0,
        "raw_mse_mean": 0.0,
        "generated_text_mean_length": 0.0,
    }
    total_samples = 0
    num_batches = len(dataloader)
    for batch_index, batch in enumerate(tqdm(dataloader, desc="rl-train", leave=False)):
        activations = batch["activations"].to(device)
        expanded_activations = repeat_for_return_sequences(
            activations,
            args.num_return_sequences,
        )
        sampled = sample_av_text(
            model=av_bundle.model,
            tokenizer=av_bundle.tokenizer,
            activations=expanded_activations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        reconstructed = reconstruct_texts_with_ar(
            ar_bundle=ar_bundle,
            texts=sampled.generated_texts,
            device=device,
            max_ar_length=args.max_ar_length,
        )
        expanded_targets = repeat_for_return_sequences(
            batch["activations"],
            args.num_return_sequences,
        )
        reward_batch = reconstruction_reward(expanded_targets, reconstructed)
        rewards = reward_batch.reward.to(device)
        advantages = reward_normalizer.normalize(rewards)
        mean_logprobs, entropy = av_token_logprobs_and_entropy(
            model=av_bundle.model,
            activations=expanded_activations,
            generated_ids=sampled.generated_ids,
            token_mask=sampled.token_mask,
        )
        policy_loss = policy_loss_from_advantage(
            mean_token_logprobs=mean_logprobs,
            advantages=advantages,
        )
        sft_loss = torch.zeros((), device=device)
        if args.kl_weight > 0.0:
            sft_loss = sft_loss_for_targets(
                model=av_bundle.model,
                activations=activations,
                input_ids=batch["target_input_ids"].to(device),
                attention_mask=batch["target_attention_mask"].to(device),
            )
        entropy_loss_value = entropy.mean()
        lengths = sampled.token_mask.float().sum(dim=1).to(device)
        length_penalty = lengths.mean() / float(args.max_new_tokens)
        total_loss = (
            policy_loss
            + args.kl_weight * sft_loss
            - args.entropy_weight * entropy_loss_value
            + args.length_penalty_weight * length_penalty
        )
        (total_loss / args.gradient_accumulation_steps).backward()
        should_step = (
            (batch_index + 1) % args.gradient_accumulation_steps == 0
            or batch_index + 1 == num_batches
        )
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_samples = expanded_activations.shape[0]
        total_samples += batch_samples
        totals["policy_loss"] += policy_loss.item() * batch_samples
        totals["sft_loss"] += sft_loss.item() * batch_samples
        totals["entropy"] += entropy_loss_value.item() * batch_samples
        reward_cpu = reward_batch.reward.detach().cpu()
        totals["reward_mean"] += reward_cpu.sum().item()
        totals["reward_sq_sum"] += (reward_cpu**2).sum().item()
        totals["normalized_mse_mean"] += reward_batch.normalized_mse.sum().item()
        totals["raw_mse_mean"] += reward_batch.raw_mse.sum().item()
        totals["generated_text_mean_length"] += lengths.detach().cpu().sum().item()

    if total_samples == 0:
        raise ValueError("RL training dataloader produced no samples.")
    reward_mean = totals["reward_mean"] / total_samples
    reward_var = max(totals["reward_sq_sum"] / total_samples - reward_mean**2, 0.0)
    return {
        "policy_loss": totals["policy_loss"] / total_samples,
        "sft_loss": totals["sft_loss"] / total_samples,
        "entropy": totals["entropy"] / total_samples,
        "reward_mean": reward_mean,
        "reward_std": reward_var**0.5,
        "normalized_mse_mean": totals["normalized_mse_mean"] / total_samples,
        "raw_mse_mean": totals["raw_mse_mean"] / total_samples,
        "generated_text_mean_length": totals["generated_text_mean_length"] / total_samples,
    }


@torch.no_grad()
def evaluate_full_loop(
    *,
    av_bundle: QwenAVCheckpointBundle,
    ar_bundle: QwenARCheckpointBundle,
    validation_artifact: ActivationArtifact,
    target_text_field: str,
    fallback_text_fields: list[str],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    max_ar_length: int,
    seed: int,
    tqdm,
) -> tuple[dict[str, float], torch.Tensor, list[dict[str, Any]]]:
    examples = build_qwen_av_examples(
        artifact=validation_artifact,
        target_text_field=target_text_field,
        fallback_text_fields=fallback_text_fields,
    )
    generated_rows = generate_qwen_av_rows(
        model=av_bundle.model,
        tokenizer=av_bundle.tokenizer,
        examples=examples,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        tqdm=tqdm,
        desc="rl-validation-generate",
        log_every_batches=10,
    )
    predictions = []
    validation_starts = range(0, len(generated_rows), batch_size)
    for start in tqdm(validation_starts, desc="rl-validation-ar", leave=False):
        batch_rows = generated_rows[start : start + batch_size]
        predictions.append(
            reconstruct_texts_with_ar(
                ar_bundle=ar_bundle,
                texts=[row["generated_text"] or "." for row in batch_rows],
                device=device,
                max_ar_length=max_ar_length,
            )
        )
    reconstructed = torch.cat(predictions, dim=0)
    original = validation_artifact.activations
    summary = summarize_reconstruction(original, reconstructed)
    normalized_mse = normalized_mse_per_example(original, reconstructed)
    reward = -normalized_mse
    mean_summary = summarize_reconstruction(
        original,
        baseline_mean_reconstruction(original, original),
    )
    zero_summary = summarize_reconstruction(original, baseline_zero_reconstruction(original))
    shuffled_summary = summarize_reconstruction(
        original,
        baseline_shuffled_reconstruction(original, seed),
    )
    metrics = {
        "validation_nla_fve": summary["fve"],
        "validation_nla_mse": summary["mse"],
        "validation_nla_rmse": summary["rmse"],
        "validation_cosine_mean": summary["cosine_mean"],
        "validation_normalized_mse": normalized_mse.mean().item(),
        "validation_reward_mean": reward.mean().item(),
        "validation_mean_baseline_mse": mean_summary["mse"],
        "validation_zero_baseline_fve": zero_summary["fve"],
        "validation_shuffled_baseline_fve": shuffled_summary["fve"],
    }
    return metrics, reconstructed, generated_rows


def empty_validation_metrics() -> dict[str, float]:
    return {
        "validation_nla_fve": float("nan"),
        "validation_nla_mse": float("nan"),
        "validation_nla_rmse": float("nan"),
        "validation_cosine_mean": float("nan"),
        "validation_normalized_mse": float("nan"),
        "validation_reward_mean": float("nan"),
        "validation_mean_baseline_mse": float("nan"),
        "validation_zero_baseline_fve": float("nan"),
        "validation_shuffled_baseline_fve": float("nan"),
    }


def metric_row(
    *,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "policy_loss": train_metrics["policy_loss"],
        "sft_loss": train_metrics["sft_loss"],
        "entropy": train_metrics["entropy"],
        "reward_mean": train_metrics["reward_mean"],
        "reward_std": train_metrics["reward_std"],
        "normalized_mse_mean": train_metrics["normalized_mse_mean"],
        "raw_mse_mean": train_metrics["raw_mse_mean"],
        "generated_text_mean_length": train_metrics["generated_text_mean_length"],
        "validation_nla_fve": validation_metrics["validation_nla_fve"],
        "validation_normalized_mse": validation_metrics["validation_normalized_mse"],
        "validation_reward_mean": validation_metrics["validation_reward_mean"],
        "is_best": is_best,
    }


def validation_per_example_rows(
    *,
    metadata_rows: list[dict],
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    squared_errors = per_example_squared_error(original, reconstructed)
    l2_errors = per_example_l2_error(original, reconstructed)
    cosine = per_example_cosine_similarity(original, reconstructed)
    normalized_mse = normalized_mse_per_example(original, reconstructed)
    rows = []
    for index, metadata in enumerate(metadata_rows):
        rows.append(
            {
                "activation_index": int(metadata.get("activation_index", index)),
                "example_id": metadata.get("example_id"),
                "squared_error": squared_errors[index].item(),
                "l2_error": l2_errors[index].item(),
                "cosine_similarity": cosine[index].item(),
                "normalized_mse": normalized_mse[index].item(),
                "reward": -normalized_mse[index].item(),
                "generated_text": generated_rows[index]["generated_text"],
            }
        )
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in METRIC_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def lora_settings_from_config(config: dict[str, Any]) -> LoraSettings:
    lora = config.get("lora", {})
    return LoraSettings(
        r=int(lora.get("r", 0)),
        alpha=int(lora.get("alpha", 16)),
        dropout=float(lora.get("dropout", 0.0)),
        target_modules=tuple(lora.get("target_modules", ())),
    )


def save_rl_av_checkpoint(
    *,
    output_dir: Path,
    av_bundle: QwenAVCheckpointBundle,
    ar_bundle: QwenARCheckpointBundle,
    args: argparse.Namespace,
    epoch: int,
    validation_metrics: dict[str, float],
) -> dict[str, str]:
    output_files = {
        "activation_projection": "activation_projection.pt",
        "tokenizer": "tokenizer",
    }
    av_bundle.tokenizer.save_pretrained(output_dir / output_files["tokenizer"])
    torch.save(
        av_bundle.model.activation_projection.state_dict(),
        output_dir / output_files["activation_projection"],
    )
    lora_enabled = av_bundle.config.get("lora", {}).get("enabled", False)
    if lora_enabled:
        output_files["qwen_adapter"] = "qwen_av_rl_adapter"
        output_files["qwen_av_rl_adapter"] = "qwen_av_rl_adapter"
        av_bundle.model.qwen_model.save_pretrained(output_dir / output_files["qwen_adapter"])
    else:
        state_path = output_dir / "qwen_av_model_state.pt"
        torch.save(av_bundle.model.qwen_model.state_dict(), state_path)
        output_files["qwen_model_state"] = state_path.name
    config = qwen_checkpoint_metadata(
        component="qwen_av",
        model_name_or_path=av_bundle.config["model_name_or_path"],
        activation_dim=int(av_bundle.config["activation_dim"]),
        dtype=args.dtype,
        lora_settings=lora_settings_from_config(av_bundle.config),
        extra_config={
            "target_text_field": args.target_text_field,
            "fallback_text_fields": args.fallback_text_fields,
            "max_new_tokens": args.max_new_tokens,
            "max_ar_length": args.max_ar_length,
            "training_stage": "phase11_av_reward_rl",
            "source_joint_checkpoint_dir": args.joint_checkpoint_dir,
            "source_qwen_av_checkpoint_dir": args.qwen_av_checkpoint_dir,
            "source_qwen_ar_checkpoint_dir": args.qwen_ar_checkpoint_dir,
            "ar_reference": {
                "schema_version": ar_bundle.checkpoint.get("schema_version"),
                "epoch": ar_bundle.checkpoint.get("epoch"),
                "config": ar_bundle.config,
            },
        },
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "output_files": output_files,
    }
    torch.save(payload, output_dir / "model.pt")
    output_files["model"] = "model.pt"
    return output_files


def checkpoint_summary(bundle: QwenAVCheckpointBundle | QwenARCheckpointBundle) -> dict[str, Any]:
    return {
        "schema_version": bundle.checkpoint.get("schema_version"),
        "epoch": bundle.checkpoint.get("epoch"),
        "config": bundle.config,
        "validation_loss": bundle.checkpoint.get("validation_loss"),
        "validation_metrics": bundle.checkpoint.get("validation_metrics"),
    }


def build_manifest_payload(
    *,
    args: argparse.Namespace | dict[str, Any],
    train_artifact: ActivationArtifact,
    validation_artifact: ActivationArtifact,
    av_checkpoint_summary: dict[str, Any],
    ar_checkpoint_summary: dict[str, Any],
    best_epoch: int,
    best_metrics: dict[str, float],
    output_files: dict[str, str],
) -> dict[str, Any]:
    cli_args = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": cli_args,
        "activation_dir": cli_args["activation_dir"],
        "validation_activation_dir": cli_args["validation_activation_dir"],
        "output_dir": cli_args["output_dir"],
        "checkpoint_mode": (
            "joint" if cli_args.get("joint_checkpoint_dir") else "split"
        ),
        "joint_checkpoint_dir": cli_args.get("joint_checkpoint_dir"),
        "qwen_av_checkpoint_dir": cli_args.get("qwen_av_checkpoint_dir"),
        "qwen_ar_checkpoint_dir": cli_args.get("qwen_ar_checkpoint_dir"),
        "reward": "negative_l2_normalized_mse",
        "reward_normalization": cli_args["reward_normalization"],
        "sampling": {
            "temperature": cli_args["temperature"],
            "top_p": cli_args["top_p"],
            "num_return_sequences": cli_args["num_return_sequences"],
            "max_new_tokens": cli_args["max_new_tokens"],
        },
        "regularization": {
            "kl_weight": cli_args["kl_weight"],
            "entropy_weight": cli_args["entropy_weight"],
            "length_penalty_weight": cli_args["length_penalty_weight"],
        },
        "train_count": train_artifact.num_examples,
        "validation_count": validation_artifact.num_examples,
        "activation_dim": train_artifact.activation_dim,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "qwen_av_checkpoint_summary": av_checkpoint_summary,
        "qwen_ar_checkpoint_summary": ar_checkpoint_summary,
        "output_files": output_files,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    fallback_fields = parse_fallback_fields(args.fallback_text_fields)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    print_section(1, 10, "Loading activation artifacts")
    train_artifact = subset_artifact(
        load_activation_artifact(Path(args.activation_dir)),
        args.limit_train,
    )
    validation_artifact = subset_artifact(
        load_activation_artifact(Path(args.validation_activation_dir)),
        args.limit_validation,
    )
    if train_artifact.activation_dim != validation_artifact.activation_dim:
        raise ValueError(
            f"Validation dim {validation_artifact.activation_dim} does not match "
            f"train dim {train_artifact.activation_dim}."
        )
    prepare_output_dir(output_dir, args.overwrite)
    print(f"Train activations: {tuple(train_artifact.activations.shape)}")
    print(f"Validation activations: {tuple(validation_artifact.activations.shape)}")

    print_section(2, 10, "Loading Qwen AV/AR checkpoints")
    av_bundle, ar_bundle, joint_bundle = load_phase11_checkpoints(args=args, device=device)
    if int(av_bundle.config["activation_dim"]) != train_artifact.activation_dim:
        raise ValueError("AV checkpoint activation_dim does not match train artifact.")
    if int(ar_bundle.config["activation_dim"]) != train_artifact.activation_dim:
        raise ValueError("AR checkpoint activation_dim does not match train artifact.")
    freeze_module(ar_bundle.model)
    ensure_tokenizer_padding(av_bundle.tokenizer)
    ensure_tokenizer_padding(ar_bundle.tokenizer)
    av_bundle.model.train()
    print(f"AV model: {av_bundle.config['model_name_or_path']}")
    print(f"AR model: {ar_bundle.config['model_name_or_path']}")
    print(f"Joint checkpoint loaded: {joint_bundle is not None}")
    print(f"AV parameters: {qwen_trainable_parameter_summary(av_bundle.model)}")

    print_section(3, 10, "Building RL datasets")
    train_examples = build_rl_examples(
        artifact=train_artifact,
        target_text_field=args.target_text_field,
        fallback_text_fields=fallback_fields,
    )
    collate_fn = make_collate_fn(av_bundle.tokenizer, args.max_new_tokens)
    train_loader = DataLoader(
        RLDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in av_bundle.model.parameters() if parameter.requires_grad],
        lr=args.learning_rate_av,
    )
    reward_normalizer = RewardNormalizer(args.reward_normalization)
    tqdm = import_tqdm()

    print_section(4, 10, "Reward RL training")
    metrics_rows = []
    best_reward = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_predictions: torch.Tensor | None = None
    best_generated_rows: list[dict[str, Any]] | None = None
    output_files: dict[str, str] = {}
    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = train_one_epoch(
            av_bundle=av_bundle,
            ar_bundle=ar_bundle,
            dataloader=train_loader,
            optimizer=optimizer,
            reward_normalizer=reward_normalizer,
            args=args,
            device=device,
            tqdm=tqdm,
        )
        validation_metrics = empty_validation_metrics()
        is_best = False
        should_evaluate = args.eval_every_epoch or epoch == args.epochs
        if should_evaluate:
            validation_metrics, predictions, generated_rows = evaluate_full_loop(
                av_bundle=av_bundle,
                ar_bundle=ar_bundle,
                validation_artifact=validation_artifact,
                target_text_field=args.target_text_field,
                fallback_text_fields=fallback_fields,
                device=device,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_ar_length=args.max_ar_length,
                seed=args.seed,
                tqdm=tqdm,
            )
            is_best = validation_metrics["validation_reward_mean"] > best_reward
            if is_best:
                best_reward = validation_metrics["validation_reward_mean"]
                best_epoch = epoch
                best_metrics = validation_metrics
                best_predictions = predictions
                best_generated_rows = generated_rows
                output_files = save_rl_av_checkpoint(
                    output_dir=output_dir,
                    av_bundle=av_bundle,
                    ar_bundle=ar_bundle,
                    args=args,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                )
        metrics_rows.append(
            metric_row(
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                is_best=is_best,
            )
        )
        print(
            f"epoch {epoch:03d}: reward={train_metrics['reward_mean']:.6f}, "
            f"policy_loss={train_metrics['policy_loss']:.6f}, "
            f"validation_reward={validation_metrics['validation_reward_mean']:.6f}, "
            f"elapsed={time.perf_counter() - start_time:.1f}s",
            flush=True,
        )

    if best_metrics is None or best_predictions is None or best_generated_rows is None:
        raise ValueError("RL training finished without an evaluated best checkpoint.")

    print_section(5, 10, "Writing metrics")
    metrics_path = output_dir / "train_rl_metrics.csv"
    write_metrics_csv(metrics_path, metrics_rows)
    output_files["train_rl_metrics"] = metrics_path.name

    print_section(6, 10, "Writing validation artifacts")
    generations_path = output_dir / "validation_generated_explanations.jsonl"
    predictions_path = output_dir / "validation_predictions.pt"
    targets_path = output_dir / "validation_targets.pt"
    per_example_path = output_dir / "validation_per_example_metrics.jsonl"
    write_jsonl(generations_path, best_generated_rows)
    torch.save(best_predictions, predictions_path)
    torch.save(validation_artifact.activations, targets_path)
    write_jsonl(
        per_example_path,
        validation_per_example_rows(
            metadata_rows=validation_artifact.metadata_rows,
            original=validation_artifact.activations,
            reconstructed=best_predictions,
            generated_rows=best_generated_rows,
        ),
    )
    output_files.update(
        {
            "validation_generated_explanations": generations_path.name,
            "validation_predictions": predictions_path.name,
            "validation_targets": targets_path.name,
            "validation_per_example_metrics": per_example_path.name,
        }
    )

    print_section(7, 10, "Writing manifest")
    manifest_path = output_dir / "train_qwen_av_reward_rl_manifest.json"
    manifest = build_manifest_payload(
        args=args,
        train_artifact=train_artifact,
        validation_artifact=validation_artifact,
        av_checkpoint_summary=checkpoint_summary(av_bundle),
        ar_checkpoint_summary=checkpoint_summary(ar_bundle),
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        output_files={**output_files, "manifest": manifest_path.name},
    )
    write_json(manifest_path, manifest)
    output_files["manifest"] = manifest_path.name

    print_section(8, 10, "Output files")
    for filename in output_files.values():
        print(f"Wrote {output_dir / filename}")

    print_section(9, 10, "Best validation")
    print(json.dumps(best_metrics, indent=2), flush=True)

    print_section(10, 10, "Result")
    print("SUCCESS: Phase 11 Qwen AV reward RL completed successfully.")


if __name__ == "__main__":
    main()
