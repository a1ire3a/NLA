"""Train a Qwen-based aligned Activation Reconstructor with LoRA."""

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
    summarize_reconstruction,
)
from nla_code_interp.qwen_models import (  # noqa: E402
    DEFAULT_QWEN_MODEL,
    LoraSettings,
    QwenActivationReconstructor,
    apply_lora,
    dtype_from_name,
    load_qwen_causal_lm,
    qwen_checkpoint_metadata,
    qwen_trainable_parameter_summary,
)
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.train_ar import (  # noqa: E402
    ActivationArtifact,
    TargetTransform,
    build_text_examples,
    ensure_tokenizer_padding,
    import_tqdm,
    import_transformers,
    load_activation_artifact,
    resolve_device,
    text_field_counts,
    text_length_summary,
)


SCHEMA_VERSION = "phase10a_qwen_ar_lora_v1"
TEXT_FIELDS = ("reference_description", "prompt", "code")
TARGET_TRANSFORMS = ("raw", "center", "standardize")
METRIC_COLUMNS = (
    "epoch",
    "train_mse_loss",
    "validation_fve",
    "validation_mse",
    "validation_rmse",
    "validation_cosine_mean",
    "validation_train_mean_baseline_fve",
    "is_best",
)


@dataclass(frozen=True)
class TextTargetExample:
    text: str
    target: torch.Tensor
    metadata: dict[str, Any]


class QwenARDataset(Dataset):
    def __init__(self, examples: list[TextTargetExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TextTargetExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA text-to-activation AR.")
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--validation_activation_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--text_field", choices=TEXT_FIELDS, default="reference_description")
    parser.add_argument("--fallback_text_fields", default="prompt,code")
    parser.add_argument(
        "--target_transform",
        choices=TARGET_TRANSFORMS,
        default="standardize",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def parse_fallback_fields(value: str) -> list[str]:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    invalid = [field for field in fields if field not in TEXT_FIELDS]
    if invalid:
        raise ValueError(f"Invalid fallback text field(s): {invalid}")
    return fields


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")
    if args.max_length <= 0:
        raise ValueError(f"max_length must be positive, got {args.max_length}")
    if args.lora_r < 0:
        raise ValueError(f"lora_r must be non-negative, got {args.lora_r}")
    if args.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {args.lora_alpha}")
    if args.lora_dropout < 0.0 or args.lora_dropout >= 1.0:
        raise ValueError(f"lora_dropout must be in [0, 1), got {args.lora_dropout}")
    for name in ("limit_train", "limit_validation"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when set, got {value}")


def subset_artifact(artifact: ActivationArtifact, limit: int | None) -> ActivationArtifact:
    if limit is None:
        return artifact
    if limit > artifact.num_examples:
        raise ValueError(f"limit={limit} exceeds available examples {artifact.num_examples}")
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


def build_examples(
    *,
    artifact: ActivationArtifact,
    transformed_targets: torch.Tensor,
    text_field: str,
    fallback_text_fields: list[str],
) -> list[TextTargetExample]:
    text_examples = build_text_examples(
        artifact.metadata_rows,
        text_field=text_field,
        fallback_text_fields=fallback_text_fields,
    )
    examples = []
    for index, text_example in enumerate(text_examples):
        metadata = dict(text_example.metadata)
        metadata["qwen_ar_text"] = text_example.text
        metadata["qwen_ar_text_field"] = text_example.selected_text_field
        examples.append(
            TextTargetExample(
                text=text_example.text,
                target=transformed_targets[index],
                metadata=metadata,
            )
        )
    return examples


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: list[TextTargetExample]) -> dict[str, Any]:
        tokenized = tokenizer(
            [item.text for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(tokenized["input_ids"])
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": attention_mask,
            "targets": torch.stack([item.target for item in batch], dim=0),
            "metadata": [item.metadata for item in batch],
        }

    return collate


def move_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["targets"].to(device),
    )


def train_one_epoch(
    *,
    model: QwenActivationReconstructor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tqdm,
) -> float:
    model.train()
    loss_fn = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_examples = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        input_ids, attention_mask, targets = move_batch(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(predictions.float(), targets.float())
        loss.backward()
        optimizer.step()
        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("Training dataloader produced no examples.")
    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    *,
    model: QwenActivationReconstructor,
    dataloader: DataLoader,
    device: torch.device,
    target_transform: TargetTransform,
    original_validation_targets: torch.Tensor,
    original_train_targets: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, list[dict]]:
    model.eval()
    transformed_batches = []
    metadata_rows = []
    for batch in dataloader:
        input_ids, attention_mask, _targets = move_batch(batch, device=device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        transformed_batches.append(predictions.detach().cpu().float())
        metadata_rows.extend(batch["metadata"])
    if not transformed_batches:
        raise ValueError("Validation dataloader produced no examples.")
    transformed_predictions = torch.cat(transformed_batches, dim=0)
    original_predictions = target_transform.inverse_transform(transformed_predictions)
    summary = summarize_reconstruction(original_validation_targets, original_predictions)
    train_mean = baseline_mean_reconstruction(
        original_train_targets,
        original_validation_targets,
    )
    train_mean_summary = summarize_reconstruction(original_validation_targets, train_mean)
    metrics = {
        "fve": summary["fve"],
        "mse": summary["mse"],
        "rmse": summary["rmse"],
        "cosine_mean": summary["cosine_mean"],
        "validation_train_mean_baseline_fve": train_mean_summary["fve"],
    }
    return metrics, original_predictions, metadata_rows


def metric_row(
    *,
    epoch: int,
    train_loss: float,
    validation_metrics: dict[str, float],
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_mse_loss": train_loss,
        "validation_fve": validation_metrics["fve"],
        "validation_mse": validation_metrics["mse"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_cosine_mean": validation_metrics["cosine_mean"],
        "validation_train_mean_baseline_fve": validation_metrics[
            "validation_train_mean_baseline_fve"
        ],
        "is_best": is_best,
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in METRIC_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_qwen_parts(
    *,
    model: QwenActivationReconstructor,
    tokenizer,
    output_dir: Path,
    lora_settings: LoraSettings,
) -> dict[str, str]:
    files = {"projection_head": "projection_head.pt", "tokenizer": "tokenizer"}
    tokenizer.save_pretrained(output_dir / "tokenizer")
    torch.save(model.projection.state_dict(), output_dir / files["projection_head"])
    if lora_settings.enabled:
        adapter_dir = output_dir / "qwen_adapter"
        model.qwen_model.save_pretrained(adapter_dir)
        files["qwen_adapter"] = "qwen_adapter"
    else:
        state_path = output_dir / "qwen_model_state.pt"
        torch.save(model.qwen_model.state_dict(), state_path)
        files["qwen_model_state"] = state_path.name
    return files


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    activation_dim: int,
    lora_settings: LoraSettings,
    target_transform: TargetTransform,
    epoch: int,
    validation_metrics: dict[str, float],
    output_files: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": qwen_checkpoint_metadata(
            component="qwen_ar",
            model_name_or_path=args.model_name_or_path,
            activation_dim=activation_dim,
            dtype=args.dtype,
            lora_settings=lora_settings,
            extra_config={
                "text_field": args.text_field,
                "fallback_text_fields": args.fallback_text_fields,
                "target_transform": args.target_transform,
                "max_length": args.max_length,
                "pooling": "final",
            },
        ),
        "target_transform_state": target_transform.state_dict_for_checkpoint(),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "output_files": output_files,
    }


def save_checkpoint(
    *,
    output_dir: Path,
    model: QwenActivationReconstructor,
    tokenizer,
    args: argparse.Namespace,
    activation_dim: int,
    lora_settings: LoraSettings,
    target_transform: TargetTransform,
    epoch: int,
    validation_metrics: dict[str, float],
) -> dict[str, str]:
    output_files = save_qwen_parts(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        lora_settings=lora_settings,
    )
    payload = checkpoint_payload(
        args=args,
        activation_dim=activation_dim,
        lora_settings=lora_settings,
        target_transform=target_transform,
        epoch=epoch,
        validation_metrics=validation_metrics,
        output_files=output_files,
    )
    torch.save(payload, output_dir / "model.pt")
    output_files["model"] = "model.pt"
    return output_files


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    fallback_fields = parse_fallback_fields(args.fallback_text_fields)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)
    lora_settings = LoraSettings(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    print_section(1, 8, "Loading activation artifacts")
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

    print_section(2, 8, "Preparing targets and text")
    target_transform = TargetTransform.fit(args.target_transform, train_artifact.activations)
    transformed_train = target_transform.transform(train_artifact.activations)
    transformed_validation = target_transform.transform(validation_artifact.activations)
    train_examples = build_examples(
        artifact=train_artifact,
        transformed_targets=transformed_train,
        text_field=args.text_field,
        fallback_text_fields=fallback_fields,
    )
    validation_examples = build_examples(
        artifact=validation_artifact,
        transformed_targets=transformed_validation,
        text_field=args.text_field,
        fallback_text_fields=fallback_fields,
    )
    text_examples = build_text_examples(
        train_artifact.metadata_rows,
        text_field=args.text_field,
        fallback_text_fields=fallback_fields,
    )
    print(f"Text field counts: {text_field_counts(text_examples)}")
    print(f"Train text lengths: {text_length_summary(text_examples)}")

    print_section(3, 8, "Loading tokenizer and Qwen AR")
    AutoTokenizer = import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    ensure_tokenizer_padding(tokenizer)
    qwen_model = load_qwen_causal_lm(
        model_name_or_path=args.model_name_or_path,
        dtype=dtype_from_name(args.dtype),
    )
    qwen_model = apply_lora(qwen_model, lora_settings=lora_settings)
    model = QwenActivationReconstructor(
        qwen_model=qwen_model,
        activation_dim=train_artifact.activation_dim,
        pooling="final",
    ).to(device)
    print(f"Model: {args.model_name_or_path}")
    print(f"LoRA: {lora_settings.as_dict()}")
    print(f"Parameters: {qwen_trainable_parameter_summary(model)}")
    print(f"Device: {device}")

    print_section(4, 8, "Training setup")
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        QwenARDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        QwenARDataset(validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    tqdm = import_tqdm()

    print_section(5, 8, "Training")
    metrics_rows = []
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_predictions: torch.Tensor | None = None
    best_metadata_rows: list[dict] | None = None
    best_fve = float("-inf")
    output_files: dict[str, str] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            tqdm=tqdm,
        )
        validation_metrics, predictions, metadata_rows = evaluate(
            model=model,
            dataloader=validation_loader,
            device=device,
            target_transform=target_transform,
            original_validation_targets=validation_artifact.activations,
            original_train_targets=train_artifact.activations,
        )
        is_best = validation_metrics["fve"] > best_fve
        if is_best:
            best_fve = validation_metrics["fve"]
            best_epoch = epoch
            best_metrics = validation_metrics
            best_predictions = predictions
            best_metadata_rows = metadata_rows
            output_files = save_checkpoint(
                output_dir=output_dir,
                model=model,
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
                train_loss=train_loss,
                validation_metrics=validation_metrics,
                is_best=is_best,
            )
        )
        print(
            f"epoch {epoch:03d}: train_mse={train_loss:.6f}, "
            f"validation_fve={validation_metrics['fve']:.6f}, "
            f"validation_mse={validation_metrics['mse']:.6f}"
        )

    if best_metrics is None or best_predictions is None or best_metadata_rows is None:
        raise ValueError("Training finished without a best checkpoint.")

    print_section(6, 8, "Writing validation artifacts")
    metrics_path = output_dir / "training_metrics.csv"
    predictions_path = output_dir / "validation_predictions.pt"
    targets_path = output_dir / "validation_targets.pt"
    metadata_path = output_dir / "validation_metadata.jsonl"
    write_metrics_csv(metrics_path, metrics_rows)
    torch.save(best_predictions, predictions_path)
    torch.save(validation_artifact.activations, targets_path)
    write_jsonl(metadata_path, best_metadata_rows)
    output_files.update(
        {
            "training_metrics": metrics_path.name,
            "validation_predictions": predictions_path.name,
            "validation_targets": targets_path.name,
            "validation_metadata": metadata_path.name,
        }
    )

    print_section(7, 8, "Writing manifest")
    manifest_path = output_dir / "train_qwen_ar_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "activation_dir": args.activation_dir,
        "validation_activation_dir": args.validation_activation_dir,
        "output_dir": args.output_dir,
        "model_name_or_path": args.model_name_or_path,
        "text_field": args.text_field,
        "fallback_text_fields": fallback_fields,
        "target_transform": args.target_transform,
        "target_transform_state": target_transform.state_dict_for_manifest(),
        "lora": lora_settings.as_dict(),
        "dtype": args.dtype,
        "train_count": train_artifact.num_examples,
        "validation_count": validation_artifact.num_examples,
        "activation_dim": train_artifact.activation_dim,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "output_files": output_files,
    }
    write_json(manifest_path, manifest)
    output_files["manifest"] = manifest_path.name
    for filename in output_files.values():
        print(f"Wrote {output_dir / filename}")

    print_section(8, 8, "Result")
    print("SUCCESS: Phase 10a Qwen AR training completed successfully.")


if __name__ == "__main__":
    main()
