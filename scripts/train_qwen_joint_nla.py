"""Train final aligned Qwen AV and AR in one coordinated NLA loop."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
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
from nla_code_interp.qwen_models import (  # noqa: E402
    LoraSettings,
    QwenActivationReconstructor,
    QwenActivationVerbalizer,
    apply_lora,
    dtype_from_name,
    load_qwen_causal_lm,
    qwen_trainable_parameter_summary,
)
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.train_ar import (  # noqa: E402
    ActivationArtifact,
    TargetTransform,
    ensure_tokenizer_padding,
    import_tqdm,
    import_transformers,
    load_activation_artifact,
    resolve_device,
)
from scripts.train_av import (  # noqa: E402
    build_av_examples as build_base_av_examples,
    target_field_counts,
    text_length_summary,
)
from scripts.train_qwen_av import (  # noqa: E402
    QwenAVDataset,
    build_examples as build_qwen_av_examples,
    generate_rows as generate_qwen_av_rows,
    make_collate_fn as make_qwen_av_collate_fn,
    move_batch as move_qwen_av_batch,
    parse_fallback_fields,
)


SCHEMA_VERSION = "phase10d_qwen_joint_nla_v1"
TARGET_TEXT_FIELDS = ("reference_description", "prompt", "code")
TARGET_TRANSFORMS = ("raw", "center", "standardize")
METRIC_COLUMNS = (
    "epoch",
    "av_train_loss",
    "ar_generated_train_mse",
    "ar_anchor_train_mse",
    "validation_nla_fve",
    "validation_nla_mse",
    "validation_nla_rmse",
    "validation_cosine_mean",
    "validation_mean_baseline_mse",
    "validation_zero_baseline_fve",
    "validation_shuffled_baseline_fve",
    "is_best",
)


@dataclass(frozen=True)
class GeneratedAnchorExample:
    generated_text: str
    anchor_text: str
    target: torch.Tensor
    metadata: dict[str, Any]


class GeneratedAnchorARDataset(Dataset):
    def __init__(self, examples: list[GeneratedAnchorExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GeneratedAnchorExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final Qwen AV/AR with alternating aligned NLA updates."
    )
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--validation_activation_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    )
    parser.add_argument(
        "--target_text_field",
        choices=TARGET_TEXT_FIELDS,
        default="reference_description",
    )
    parser.add_argument("--fallback_text_fields", default="prompt,code")
    parser.add_argument(
        "--target_transform",
        choices=TARGET_TRANSFORMS,
        default="standardize",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate_av", type=float, default=1e-4)
    parser.add_argument("--learning_rate_ar", type=float, default=1e-4)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_ar_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
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
    parser.add_argument("--av_loss_weight", type=float, default=1.0)
    parser.add_argument("--ar_generated_loss_weight", type=float, default=1.0)
    parser.add_argument("--ar_anchor_loss_weight", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_target_length",
        "max_ar_length",
        "max_new_tokens",
        "lora_alpha",
    )
    for name in positive_ints:
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name in ("limit_train", "limit_validation"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when set, got {value}")
    if args.lora_r < 0:
        raise ValueError(f"lora_r must be non-negative, got {args.lora_r}")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError(f"lora_dropout must be in [0, 1), got {args.lora_dropout}")
    for name in ("learning_rate_av", "learning_rate_ar"):
        value = getattr(args, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name in ("av_loss_weight", "ar_generated_loss_weight", "ar_anchor_loss_weight"):
        value = getattr(args, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if args.ar_generated_loss_weight == 0.0 and args.ar_anchor_loss_weight == 0.0:
        raise ValueError("At least one AR loss weight must be positive.")


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


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


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is non-empty: {output_dir}. "
                "Pass --overwrite to replace it intentionally."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def make_joint_ar_collate_fn(tokenizer, max_length: int):
    def collate(batch: list[GeneratedAnchorExample]) -> dict[str, Any]:
        generated = tokenizer(
            [item.generated_text for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        anchor = tokenizer(
            [item.anchor_text for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "generated_input_ids": generated["input_ids"],
            "generated_attention_mask": generated.get(
                "attention_mask",
                torch.ones_like(generated["input_ids"]),
            ),
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor.get(
                "attention_mask",
                torch.ones_like(anchor["input_ids"]),
            ),
            "targets": torch.stack([item.target for item in batch], dim=0),
            "metadata": [item.metadata for item in batch],
        }

    return collate


def build_generated_anchor_examples(
    *,
    artifact: ActivationArtifact,
    generated_rows: list[dict[str, Any]],
    transformed_targets: torch.Tensor,
    target_text_field: str,
    fallback_text_fields: list[str],
) -> list[GeneratedAnchorExample]:
    if len(generated_rows) != artifact.num_examples:
        raise ValueError(
            f"Generated row count {len(generated_rows)} does not match "
            f"artifact rows {artifact.num_examples}."
        )
    if transformed_targets.shape[0] != artifact.num_examples:
        raise ValueError(
            f"Target rows {transformed_targets.shape[0]} do not match "
            f"artifact rows {artifact.num_examples}."
        )
    anchor_examples = build_qwen_av_examples(
        artifact=artifact,
        target_text_field=target_text_field,
        fallback_text_fields=fallback_text_fields,
    )
    examples = []
    for index, row in enumerate(generated_rows):
        metadata = dict(artifact.metadata_rows[index])
        metadata.update(
            {
                "qwen_joint_generated_text": row["generated_text"],
                "qwen_joint_anchor_text": anchor_examples[index].target_text,
                "qwen_joint_anchor_field": anchor_examples[index].metadata[
                    "qwen_av_target_field"
                ],
            }
        )
        examples.append(
            GeneratedAnchorExample(
                generated_text=row["generated_text"],
                anchor_text=anchor_examples[index].target_text,
                target=transformed_targets[index],
                metadata=metadata,
            )
        )
    return examples


def metric_row(
    *,
    epoch: int,
    av_train_loss: float,
    ar_generated_train_mse: float,
    ar_anchor_train_mse: float,
    validation_metrics: dict[str, float],
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "av_train_loss": av_train_loss,
        "ar_generated_train_mse": ar_generated_train_mse,
        "ar_anchor_train_mse": ar_anchor_train_mse,
        "validation_nla_fve": validation_metrics["validation_nla_fve"],
        "validation_nla_mse": validation_metrics["validation_nla_mse"],
        "validation_nla_rmse": validation_metrics["validation_nla_rmse"],
        "validation_cosine_mean": validation_metrics["validation_cosine_mean"],
        "validation_mean_baseline_mse": validation_metrics[
            "validation_mean_baseline_mse"
        ],
        "validation_zero_baseline_fve": validation_metrics[
            "validation_zero_baseline_fve"
        ],
        "validation_shuffled_baseline_fve": validation_metrics[
            "validation_shuffled_baseline_fve"
        ],
        "is_best": is_best,
    }


def empty_validation_metrics() -> dict[str, float]:
    return {
        "validation_nla_fve": float("nan"),
        "validation_nla_mse": float("nan"),
        "validation_nla_rmse": float("nan"),
        "validation_cosine_mean": float("nan"),
        "validation_mean_baseline_mse": float("nan"),
        "validation_zero_baseline_fve": float("nan"),
        "validation_shuffled_baseline_fve": float("nan"),
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in METRIC_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def optimizer_step_if_needed(
    *,
    optimizer: torch.optim.Optimizer,
    batch_index: int,
    num_batches: int,
    gradient_accumulation_steps: int,
) -> None:
    should_step = (
        (batch_index + 1) % gradient_accumulation_steps == 0
        or batch_index + 1 == num_batches
    )
    if should_step:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def train_av_one_epoch(
    *,
    model: QwenActivationVerbalizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int,
    loss_weight: float,
    tqdm,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    num_batches = len(dataloader)
    for batch_index, batch in enumerate(tqdm(dataloader, desc="train-av", leave=False)):
        activations, input_ids, attention_mask = move_qwen_av_batch(batch, device=device)
        outputs = model(
            activations=activations,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        if outputs.loss is None:
            raise ValueError("Qwen AV did not return a training loss.")
        if loss_weight > 0.0:
            scaled_loss = outputs.loss * loss_weight / gradient_accumulation_steps
            scaled_loss.backward()
            optimizer_step_if_needed(
                optimizer=optimizer,
                batch_index=batch_index,
                num_batches=num_batches,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        batch_size = activations.shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("AV training dataloader produced no examples.")
    return total_loss / total_examples


def move_ar_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["generated_input_ids"].to(device),
        batch["generated_attention_mask"].to(device),
        batch["anchor_input_ids"].to(device),
        batch["anchor_attention_mask"].to(device),
        batch["targets"].to(device),
    )


def train_ar_one_epoch(
    *,
    model: QwenActivationReconstructor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int,
    generated_loss_weight: float,
    anchor_loss_weight: float,
    tqdm,
) -> dict[str, float]:
    model.train()
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer.zero_grad(set_to_none=True)
    generated_loss_sum = 0.0
    anchor_loss_sum = 0.0
    total_examples = 0
    num_batches = len(dataloader)
    for batch_index, batch in enumerate(tqdm(dataloader, desc="train-ar", leave=False)):
        (
            generated_input_ids,
            generated_attention_mask,
            anchor_input_ids,
            anchor_attention_mask,
            targets,
        ) = move_ar_batch(batch, device=device)
        total_loss = torch.zeros((), device=device)
        generated_loss = torch.zeros((), device=device)
        anchor_loss = torch.zeros((), device=device)
        if generated_loss_weight > 0.0:
            generated_predictions = model(
                input_ids=generated_input_ids,
                attention_mask=generated_attention_mask,
            )
            generated_loss = loss_fn(generated_predictions.float(), targets.float())
            total_loss = total_loss + generated_loss_weight * generated_loss
        if anchor_loss_weight > 0.0:
            anchor_predictions = model(
                input_ids=anchor_input_ids,
                attention_mask=anchor_attention_mask,
            )
            anchor_loss = loss_fn(anchor_predictions.float(), targets.float())
            total_loss = total_loss + anchor_loss_weight * anchor_loss
        (total_loss / gradient_accumulation_steps).backward()
        optimizer_step_if_needed(
            optimizer=optimizer,
            batch_index=batch_index,
            num_batches=num_batches,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        batch_size = targets.shape[0]
        generated_loss_sum += generated_loss.item() * batch_size
        anchor_loss_sum += anchor_loss.item() * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("AR training dataloader produced no examples.")
    return {
        "ar_generated_train_mse": generated_loss_sum / total_examples,
        "ar_anchor_train_mse": anchor_loss_sum / total_examples,
    }


@torch.no_grad()
def generate_for_artifact(
    *,
    model: QwenActivationVerbalizer,
    tokenizer,
    artifact: ActivationArtifact,
    target_text_field: str,
    fallback_text_fields: list[str],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    examples = build_qwen_av_examples(
        artifact=artifact,
        target_text_field=target_text_field,
        fallback_text_fields=fallback_text_fields,
    )
    return generate_qwen_av_rows(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )


@torch.no_grad()
def reconstruct_generated_text(
    *,
    model: QwenActivationReconstructor,
    tokenizer,
    target_transform: TargetTransform,
    generated_rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_ar_length: int,
) -> torch.Tensor:
    model.eval()
    prediction_batches = []
    for start in range(0, len(generated_rows), batch_size):
        batch_rows = generated_rows[start : start + batch_size]
        tokenized = tokenizer(
            [row["generated_text"] for row in batch_rows],
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
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction_batches.append(predictions.detach().cpu().float())
    if not prediction_batches:
        raise ValueError("No generated rows were available for AR reconstruction.")
    transformed = torch.cat(prediction_batches, dim=0)
    return target_transform.inverse_transform(transformed)


def validation_metric_payload(
    *,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    nla_summary = summarize_reconstruction(original, reconstructed)
    mean_summary = summarize_reconstruction(
        original,
        baseline_mean_reconstruction(original, original),
    )
    zero_summary = summarize_reconstruction(
        original,
        baseline_zero_reconstruction(original),
    )
    shuffled_summary = summarize_reconstruction(
        original,
        baseline_shuffled_reconstruction(original, seed),
    )
    return {
        "validation_nla_fve": nla_summary["fve"],
        "validation_nla_mse": nla_summary["mse"],
        "validation_nla_rmse": nla_summary["rmse"],
        "validation_cosine_mean": nla_summary["cosine_mean"],
        "validation_mean_baseline_mse": mean_summary["mse"],
        "validation_zero_baseline_fve": zero_summary["fve"],
        "validation_shuffled_baseline_fve": shuffled_summary["fve"],
    }


@torch.no_grad()
def evaluate_full_loop(
    *,
    av_model: QwenActivationVerbalizer,
    ar_model: QwenActivationReconstructor,
    tokenizer,
    validation_artifact: ActivationArtifact,
    target_transform: TargetTransform,
    target_text_field: str,
    fallback_text_fields: list[str],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    max_ar_length: int,
    seed: int,
) -> tuple[dict[str, float], torch.Tensor, list[dict[str, Any]]]:
    generated_rows = generate_for_artifact(
        model=av_model,
        tokenizer=tokenizer,
        artifact=validation_artifact,
        target_text_field=target_text_field,
        fallback_text_fields=fallback_text_fields,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    predictions = reconstruct_generated_text(
        model=ar_model,
        tokenizer=tokenizer,
        target_transform=target_transform,
        generated_rows=generated_rows,
        device=device,
        batch_size=batch_size,
        max_ar_length=max_ar_length,
    )
    if predictions.shape != validation_artifact.activations.shape:
        raise ValueError(
            f"Validation prediction shape {tuple(predictions.shape)} does not match "
            f"targets {tuple(validation_artifact.activations.shape)}."
        )
    metrics = validation_metric_payload(
        original=validation_artifact.activations,
        reconstructed=predictions,
        seed=seed,
    )
    return metrics, predictions, generated_rows


def validation_per_example_rows(
    *,
    metadata_rows: list[dict],
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(metadata_rows) != original.shape[0] or len(generated_rows) != original.shape[0]:
        raise ValueError("metadata, activations, and generated rows must align.")
    squared_errors = per_example_squared_error(original, reconstructed)
    l2_errors = per_example_l2_error(original, reconstructed)
    cosine_similarities = per_example_cosine_similarity(original, reconstructed)
    rows = []
    for index, metadata in enumerate(metadata_rows):
        rows.append(
            {
                "activation_index": int(metadata.get("activation_index", index)),
                "example_id": metadata.get("example_id"),
                "squared_error": squared_errors[index].item(),
                "l2_error": l2_errors[index].item(),
                "cosine_similarity": cosine_similarities[index].item(),
                "generated_text": generated_rows[index]["generated_text"],
            }
        )
    return rows


def checkpoint_config(
    *,
    args: argparse.Namespace,
    activation_dim: int,
    lora_settings: LoraSettings,
) -> dict[str, Any]:
    return {
        "component": "qwen_joint_nla",
        "model_name_or_path": args.model_name_or_path,
        "activation_dim": activation_dim,
        "dtype": args.dtype,
        "lora": lora_settings.as_dict(),
        "target_text_field": args.target_text_field,
        "fallback_text_fields": args.fallback_text_fields,
        "target_transform": args.target_transform,
        "max_target_length": args.max_target_length,
        "max_ar_length": args.max_ar_length,
        "max_new_tokens": args.max_new_tokens,
        "pooling": "final",
        "loss_weights": {
            "av_loss_weight": args.av_loss_weight,
            "ar_generated_loss_weight": args.ar_generated_loss_weight,
            "ar_anchor_loss_weight": args.ar_anchor_loss_weight,
        },
    }


def save_joint_checkpoint(
    *,
    output_dir: Path,
    av_model: QwenActivationVerbalizer,
    ar_model: QwenActivationReconstructor,
    tokenizer,
    args: argparse.Namespace,
    activation_dim: int,
    lora_settings: LoraSettings,
    target_transform: TargetTransform,
    epoch: int,
    validation_metrics: dict[str, float],
) -> dict[str, str]:
    output_files = {
        "activation_projection": "activation_projection.pt",
        "ar_projection_head": "ar_projection_head.pt",
        "tokenizer": "tokenizer",
        "target_transform": "target_transform.pt",
    }
    tokenizer.save_pretrained(output_dir / output_files["tokenizer"])
    torch.save(
        av_model.activation_projection.state_dict(),
        output_dir / output_files["activation_projection"],
    )
    torch.save(
        ar_model.projection.state_dict(),
        output_dir / output_files["ar_projection_head"],
    )
    if lora_settings.enabled:
        av_adapter_dir = output_dir / "qwen_av_adapter"
        ar_adapter_dir = output_dir / "qwen_ar_adapter"
        av_model.qwen_model.save_pretrained(av_adapter_dir)
        ar_model.qwen_model.save_pretrained(ar_adapter_dir)
        output_files["qwen_av_adapter"] = "qwen_av_adapter"
        output_files["qwen_ar_adapter"] = "qwen_ar_adapter"
    else:
        av_state_path = output_dir / "qwen_av_model_state.pt"
        ar_state_path = output_dir / "qwen_ar_model_state.pt"
        torch.save(av_model.qwen_model.state_dict(), av_state_path)
        torch.save(ar_model.qwen_model.state_dict(), ar_state_path)
        output_files["qwen_av_model_state"] = av_state_path.name
        output_files["qwen_ar_model_state"] = ar_state_path.name

    transform_state = target_transform.state_dict_for_checkpoint()
    torch.save(transform_state, output_dir / output_files["target_transform"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": checkpoint_config(
            args=args,
            activation_dim=activation_dim,
            lora_settings=lora_settings,
        ),
        "target_transform_state": transform_state,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "output_files": output_files,
    }
    torch.save(payload, output_dir / "model.pt")
    output_files["model"] = "model.pt"
    return output_files


def build_manifest_payload(
    *,
    args: argparse.Namespace | dict[str, Any],
    train_artifact: ActivationArtifact,
    validation_artifact: ActivationArtifact,
    target_transform: TargetTransform,
    lora_settings: LoraSettings,
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
        "model_name_or_path": cli_args["model_name_or_path"],
        "target_text_field": cli_args["target_text_field"],
        "fallback_text_fields": cli_args["fallback_text_fields"],
        "target_transform": target_transform.state_dict_for_manifest(),
        "lora": lora_settings.as_dict(),
        "loss_weights": {
            "av_loss_weight": cli_args["av_loss_weight"],
            "ar_generated_loss_weight": cli_args["ar_generated_loss_weight"],
            "ar_anchor_loss_weight": cli_args["ar_anchor_loss_weight"],
        },
        "train_count": train_artifact.num_examples,
        "validation_count": validation_artifact.num_examples,
        "activation_dim": train_artifact.activation_dim,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "output_files": output_files,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    fallback_fields = parse_fallback_fields(args.fallback_text_fields)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    lora_settings = LoraSettings(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

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
            f"Validation activation dim {validation_artifact.activation_dim} does not "
            f"match train dim {train_artifact.activation_dim}."
        )
    prepare_output_dir(output_dir, args.overwrite)
    print(f"Train activations: {tuple(train_artifact.activations.shape)}")
    print(f"Validation activations: {tuple(validation_artifact.activations.shape)}")

    print_section(2, 10, "Preparing text and target transform")
    target_transform = TargetTransform.fit(args.target_transform, train_artifact.activations)
    transformed_train = target_transform.transform(train_artifact.activations)
    train_av_examples = build_qwen_av_examples(
        artifact=train_artifact,
        target_text_field=args.target_text_field,
        fallback_text_fields=fallback_fields,
    )
    train_av_stats_examples = build_base_av_examples(
        train_artifact.metadata_rows,
        target_text_field=args.target_text_field,
        fallback_text_fields=fallback_fields,
    )
    print(f"Train target fields: {target_field_counts(train_av_stats_examples)}")
    print(f"Train target lengths: {text_length_summary(train_av_stats_examples)}")
    print(f"Target transform: {args.target_transform}")

    print_section(3, 10, "Loading tokenizer and Qwen models")
    AutoTokenizer = import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    ensure_tokenizer_padding(tokenizer)
    av_qwen_model = load_qwen_causal_lm(
        model_name_or_path=args.model_name_or_path,
        dtype=dtype_from_name(args.dtype),
    )
    ar_qwen_model = load_qwen_causal_lm(
        model_name_or_path=args.model_name_or_path,
        dtype=dtype_from_name(args.dtype),
    )
    av_qwen_model = apply_lora(av_qwen_model, lora_settings=lora_settings)
    ar_qwen_model = apply_lora(ar_qwen_model, lora_settings=lora_settings)
    av_model = QwenActivationVerbalizer(
        qwen_model=av_qwen_model,
        activation_dim=train_artifact.activation_dim,
    ).to(device)
    ar_model = QwenActivationReconstructor(
        qwen_model=ar_qwen_model,
        activation_dim=train_artifact.activation_dim,
        pooling="final",
    ).to(device)
    print(f"Model: {args.model_name_or_path}")
    print(f"LoRA: {lora_settings.as_dict()}")
    print(f"AV parameters: {qwen_trainable_parameter_summary(av_model)}")
    print(f"AR parameters: {qwen_trainable_parameter_summary(ar_model)}")
    print(f"Device: {device}")

    print_section(4, 10, "Building dataloaders and optimizers")
    av_collate_fn = make_qwen_av_collate_fn(tokenizer, args.max_target_length)
    av_train_loader = DataLoader(
        QwenAVDataset(train_av_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=av_collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    av_optimizer = torch.optim.AdamW(
        [parameter for parameter in av_model.parameters() if parameter.requires_grad],
        lr=args.learning_rate_av,
    )
    ar_optimizer = torch.optim.AdamW(
        [parameter for parameter in ar_model.parameters() if parameter.requires_grad],
        lr=args.learning_rate_ar,
    )
    ar_collate_fn = make_joint_ar_collate_fn(tokenizer, args.max_ar_length)
    tqdm = import_tqdm()

    print_section(5, 10, "Joint alternating training")
    metrics_rows = []
    best_fve = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_predictions: torch.Tensor | None = None
    best_validation_generated_rows: list[dict[str, Any]] | None = None
    best_train_generated_rows: list[dict[str, Any]] | None = None
    output_files: dict[str, str] = {}
    for epoch in range(1, args.epochs + 1):
        av_train_loss = train_av_one_epoch(
            model=av_model,
            dataloader=av_train_loader,
            optimizer=av_optimizer,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            loss_weight=args.av_loss_weight,
            tqdm=tqdm,
        )
        train_generated_rows = generate_for_artifact(
            model=av_model,
            tokenizer=tokenizer,
            artifact=train_artifact,
            target_text_field=args.target_text_field,
            fallback_text_fields=fallback_fields,
            device=device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        ar_train_examples = build_generated_anchor_examples(
            artifact=train_artifact,
            generated_rows=train_generated_rows,
            transformed_targets=transformed_train,
            target_text_field=args.target_text_field,
            fallback_text_fields=fallback_fields,
        )
        ar_train_loader = DataLoader(
            GeneratedAnchorARDataset(ar_train_examples),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=ar_collate_fn,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        ar_losses = train_ar_one_epoch(
            model=ar_model,
            dataloader=ar_train_loader,
            optimizer=ar_optimizer,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            generated_loss_weight=args.ar_generated_loss_weight,
            anchor_loss_weight=args.ar_anchor_loss_weight,
            tqdm=tqdm,
        )

        should_evaluate = args.eval_every_epoch or epoch == args.epochs
        validation_metrics = empty_validation_metrics()
        is_best = False
        if should_evaluate:
            validation_metrics, predictions, validation_generated_rows = evaluate_full_loop(
                av_model=av_model,
                ar_model=ar_model,
                tokenizer=tokenizer,
                validation_artifact=validation_artifact,
                target_transform=target_transform,
                target_text_field=args.target_text_field,
                fallback_text_fields=fallback_fields,
                device=device,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_ar_length=args.max_ar_length,
                seed=args.seed,
            )
            is_best = validation_metrics["validation_nla_fve"] > best_fve
            if is_best:
                best_fve = validation_metrics["validation_nla_fve"]
                best_epoch = epoch
                best_metrics = validation_metrics
                best_predictions = predictions
                best_validation_generated_rows = validation_generated_rows
                best_train_generated_rows = train_generated_rows
                output_files = save_joint_checkpoint(
                    output_dir=output_dir,
                    av_model=av_model,
                    ar_model=ar_model,
                    tokenizer=tokenizer,
                    args=args,
                    activation_dim=train_artifact.activation_dim,
                    lora_settings=lora_settings,
                    target_transform=target_transform,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                )

        metrics_rows.append(
            metric_row(
                epoch=epoch,
                av_train_loss=av_train_loss,
                ar_generated_train_mse=ar_losses["ar_generated_train_mse"],
                ar_anchor_train_mse=ar_losses["ar_anchor_train_mse"],
                validation_metrics=validation_metrics,
                is_best=is_best,
            )
        )
        print(
            f"epoch {epoch:03d}: av_loss={av_train_loss:.6f}, "
            f"ar_gen_mse={ar_losses['ar_generated_train_mse']:.6f}, "
            f"ar_anchor_mse={ar_losses['ar_anchor_train_mse']:.6f}, "
            f"validation_fve={validation_metrics['validation_nla_fve']:.6f}"
        )

    if (
        best_metrics is None
        or best_predictions is None
        or best_validation_generated_rows is None
        or best_train_generated_rows is None
    ):
        raise ValueError("Training finished without an evaluated best checkpoint.")

    print_section(6, 10, "Writing generated text")
    train_generated_path = output_dir / "train_generated_explanations.jsonl"
    validation_generated_path = output_dir / "validation_generated_explanations.jsonl"
    write_jsonl(train_generated_path, best_train_generated_rows)
    write_jsonl(validation_generated_path, best_validation_generated_rows)
    output_files.update(
        {
            "train_generated_explanations": train_generated_path.name,
            "validation_generated_explanations": validation_generated_path.name,
        }
    )

    print_section(7, 10, "Writing validation artifacts")
    metrics_path = output_dir / "training_metrics.csv"
    predictions_path = output_dir / "validation_predictions.pt"
    targets_path = output_dir / "validation_targets.pt"
    per_example_path = output_dir / "validation_per_example_metrics.jsonl"
    write_metrics_csv(metrics_path, metrics_rows)
    torch.save(best_predictions, predictions_path)
    torch.save(validation_artifact.activations, targets_path)
    write_jsonl(
        per_example_path,
        validation_per_example_rows(
            metadata_rows=validation_artifact.metadata_rows,
            original=validation_artifact.activations,
            reconstructed=best_predictions,
            generated_rows=best_validation_generated_rows,
        ),
    )
    output_files.update(
        {
            "training_metrics": metrics_path.name,
            "validation_predictions": predictions_path.name,
            "validation_targets": targets_path.name,
            "validation_per_example_metrics": per_example_path.name,
        }
    )

    print_section(8, 10, "Writing manifest")
    manifest_path = output_dir / "train_qwen_joint_nla_manifest.json"
    manifest = build_manifest_payload(
        args=args,
        train_artifact=train_artifact,
        validation_artifact=validation_artifact,
        target_transform=target_transform,
        lora_settings=lora_settings,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        output_files={**output_files, "manifest": manifest_path.name},
    )
    write_json(manifest_path, manifest)
    output_files["manifest"] = manifest_path.name

    print_section(9, 10, "Output files")
    for filename in output_files.values():
        print(f"Wrote {output_dir / filename}")

    print_section(10, 10, "Result")
    print("SUCCESS: Phase 10d Qwen joint NLA training completed successfully.")


if __name__ == "__main__":
    main()
