"""Adapt Qwen AR on Qwen AV-generated explanations for reconstruction."""

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
    summarize_reconstruction,
)
from nla_code_interp.qwen_models import (  # noqa: E402
    LoraSettings,
    QwenARCheckpointBundle,
    QwenTargetTransform,
    load_qwen_ar_checkpoint,
    load_qwen_av_checkpoint,
    qwen_trainable_parameter_summary,
)
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.run_qwen_nla_loop import generate_qwen_explanation_rows  # noqa: E402
from scripts.train_ar import ActivationArtifact, import_tqdm, load_activation_artifact  # noqa: E402
from scripts.train_qwen_ar import make_collate_fn, move_batch  # noqa: E402


SCHEMA_VERSION = "phase10c_qwen_nla_reconstruction_v1"
METRIC_COLUMNS = (
    "epoch",
    "train_mse",
    "validation_fve",
    "validation_mse",
    "validation_rmse",
    "validation_cosine_mean",
    "validation_mean_baseline_fve",
    "validation_zero_baseline_fve",
    "validation_shuffled_baseline_fve",
    "is_best",
)


@dataclass(frozen=True)
class GeneratedTextTargetExample:
    text: str
    target: torch.Tensor
    metadata: dict[str, Any]


class GeneratedTextARDataset(Dataset):
    def __init__(self, examples: list[GeneratedTextTargetExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GeneratedTextTargetExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt Qwen AR on Qwen AV-generated explanations."
    )
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--validation_activation_dir", required=True)
    parser.add_argument("--qwen_av_checkpoint_dir", required=True)
    parser.add_argument("--qwen_ar_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_validation", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_ar_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("limit_train", "limit_validation"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when set, got {value}")
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {args.max_new_tokens}")
    if args.max_ar_length <= 0:
        raise ValueError(f"max_ar_length must be positive, got {args.max_ar_length}")


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def build_generated_text_examples(
    *,
    artifact: ActivationArtifact,
    generated_rows: list[dict[str, Any]],
    transformed_targets: torch.Tensor,
) -> list[GeneratedTextTargetExample]:
    if len(generated_rows) != artifact.num_examples:
        raise ValueError(
            f"Generated row count {len(generated_rows)} does not match "
            f"artifact rows {artifact.num_examples}."
        )
    if transformed_targets.shape[0] != artifact.num_examples:
        raise ValueError(
            f"Transformed target rows {transformed_targets.shape[0]} do not match "
            f"artifact rows {artifact.num_examples}."
        )
    examples = []
    for index, row in enumerate(generated_rows):
        metadata = dict(artifact.metadata_rows[index])
        metadata.update(
            {
                "qwen_nla_source_text": row["generated_text"],
                "qwen_nla_target_text": row.get("target_text"),
                "qwen_nla_generated_activation_index": row.get("activation_index"),
            }
        )
        examples.append(
            GeneratedTextTargetExample(
                text=row["generated_text"],
                target=transformed_targets[index],
                metadata=metadata,
            )
        )
    return examples


def lora_settings_from_config(config: dict[str, Any]) -> LoraSettings:
    lora = config.get("lora", {})
    return LoraSettings(
        r=int(lora.get("r", 0)),
        alpha=int(lora.get("alpha", 16)),
        dropout=float(lora.get("dropout", 0.0)),
        target_modules=tuple(lora.get("target_modules", ())),
    )


def train_one_epoch(
    *,
    model,
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
def evaluate_adapted_ar(
    *,
    ar_bundle: QwenARCheckpointBundle,
    dataloader: DataLoader,
    device: torch.device,
    original_validation_targets: torch.Tensor,
    seed: int,
) -> tuple[dict[str, float], torch.Tensor, list[dict]]:
    ar_bundle.model.eval()
    transformed_batches = []
    metadata_rows = []
    for batch in dataloader:
        input_ids, attention_mask, _targets = move_batch(batch, device=device)
        predictions = ar_bundle.model(input_ids=input_ids, attention_mask=attention_mask)
        transformed_batches.append(predictions.detach().cpu().float())
        metadata_rows.extend(batch["metadata"])
    if not transformed_batches:
        raise ValueError("Validation dataloader produced no examples.")

    transformed_predictions = torch.cat(transformed_batches, dim=0)
    original_predictions = ar_bundle.target_transform.inverse_transform(
        transformed_predictions
    )
    summary = summarize_reconstruction(original_validation_targets, original_predictions)
    mean_summary = summarize_reconstruction(
        original_validation_targets,
        baseline_mean_reconstruction(original_validation_targets, original_validation_targets),
    )
    zero_summary = summarize_reconstruction(
        original_validation_targets,
        baseline_zero_reconstruction(original_validation_targets),
    )
    shuffled_summary = summarize_reconstruction(
        original_validation_targets,
        baseline_shuffled_reconstruction(original_validation_targets, seed),
    )
    metrics = {
        "validation_fve": summary["fve"],
        "validation_mse": summary["mse"],
        "validation_rmse": summary["rmse"],
        "validation_cosine_mean": summary["cosine_mean"],
        "validation_mean_baseline_fve": mean_summary["fve"],
        "validation_zero_baseline_fve": zero_summary["fve"],
        "validation_shuffled_baseline_fve": shuffled_summary["fve"],
    }
    return metrics, original_predictions, metadata_rows


def metric_row(
    *,
    epoch: int,
    train_mse: float,
    validation_metrics: dict[str, float],
    is_best: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_mse": train_mse,
        **validation_metrics,
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


def save_adapted_qwen_ar_checkpoint(
    *,
    output_dir: Path,
    ar_bundle: QwenARCheckpointBundle,
    target_transform: QwenTargetTransform,
    epoch: int,
    validation_metrics: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, str]:
    output_files = {"projection_head": "projection_head.pt", "tokenizer": "tokenizer"}
    ar_bundle.tokenizer.save_pretrained(output_dir / output_files["tokenizer"])
    torch.save(
        ar_bundle.model.projection.state_dict(),
        output_dir / output_files["projection_head"],
    )
    lora_config = ar_bundle.config.get("lora", {})
    if lora_config.get("enabled", False):
        adapter_dir = output_dir / "qwen_adapter"
        ar_bundle.model.qwen_model.save_pretrained(adapter_dir)
        output_files["qwen_adapter"] = "qwen_adapter"
    else:
        state_path = output_dir / "qwen_model_state.pt"
        torch.save(ar_bundle.model.qwen_model.state_dict(), state_path)
        output_files["qwen_model_state"] = state_path.name

    config = dict(ar_bundle.config)
    config["max_length"] = args.max_ar_length
    config["adaptation"] = {
        "phase": "10c",
        "source_text": "qwen_av_generated_text",
        "source_qwen_ar_checkpoint_dir": args.qwen_ar_checkpoint_dir,
        "qwen_av_checkpoint_dir": args.qwen_av_checkpoint_dir,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "target_transform_state": target_transform.state_dict_for_checkpoint(),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "source_qwen_ar_checkpoint": ar_bundle.checkpoint,
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
    av_checkpoint_summary: dict[str, Any],
    ar_checkpoint_summary: dict[str, Any],
    target_transform: QwenTargetTransform,
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
        "qwen_av_checkpoint_dir": cli_args["qwen_av_checkpoint_dir"],
        "qwen_ar_checkpoint_dir": cli_args["qwen_ar_checkpoint_dir"],
        "output_dir": cli_args["output_dir"],
        "limit_train": cli_args.get("limit_train"),
        "limit_validation": cli_args.get("limit_validation"),
        "train_count": train_artifact.num_examples,
        "validation_count": validation_artifact.num_examples,
        "activation_dim": train_artifact.activation_dim,
        "target_transform": target_transform.state_dict_for_manifest(),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "qwen_av_checkpoint_summary": av_checkpoint_summary,
        "qwen_ar_checkpoint_summary": ar_checkpoint_summary,
        "output_files": output_files,
    }


def checkpoint_summary(bundle) -> dict[str, Any]:
    return {
        "schema_version": bundle.checkpoint.get("schema_version"),
        "epoch": bundle.checkpoint.get("epoch"),
        "config": bundle.config,
        "validation_loss": bundle.checkpoint.get("validation_loss"),
        "validation_metrics": bundle.checkpoint.get("validation_metrics"),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device()
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
            f"Validation activation dim {validation_artifact.activation_dim} does not "
            f"match train dim {train_artifact.activation_dim}."
        )
    prepare_output_dir(output_dir, args.overwrite)
    print(f"Train activations: {tuple(train_artifact.activations.shape)}")
    print(f"Validation activations: {tuple(validation_artifact.activations.shape)}")

    print_section(2, 10, "Loading Qwen AV checkpoint")
    av_bundle = load_qwen_av_checkpoint(
        checkpoint_dir=Path(args.qwen_av_checkpoint_dir),
        device=device,
        adapter_trainable=False,
    )
    print(f"Qwen AV model: {av_bundle.config['model_name_or_path']}")

    print_section(3, 10, "Generating train and validation explanations")
    train_generated_rows = generate_qwen_explanation_rows(
        artifact=train_artifact,
        av_bundle=av_bundle,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    validation_generated_rows = generate_qwen_explanation_rows(
        artifact=validation_artifact,
        av_bundle=av_bundle,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Generated train rows: {len(train_generated_rows)}")
    print(f"Generated validation rows: {len(validation_generated_rows)}")

    print_section(4, 10, "Loading Qwen AR checkpoint")
    ar_bundle = load_qwen_ar_checkpoint(
        checkpoint_dir=Path(args.qwen_ar_checkpoint_dir),
        device=device,
        adapter_trainable=True,
    )
    if int(ar_bundle.config["activation_dim"]) != train_artifact.activation_dim:
        raise ValueError(
            f"Qwen AR activation_dim={ar_bundle.config['activation_dim']} does not "
            f"match artifact dim {train_artifact.activation_dim}."
        )
    print(f"Qwen AR model: {ar_bundle.config['model_name_or_path']}")
    print(f"Qwen AR target transform: {ar_bundle.target_transform.name}")
    print(f"Parameters: {qwen_trainable_parameter_summary(ar_bundle.model)}")

    print_section(5, 10, "Building generated-text AR datasets")
    transformed_train = ar_bundle.target_transform.transform(train_artifact.activations)
    transformed_validation = ar_bundle.target_transform.transform(
        validation_artifact.activations
    )
    train_examples = build_generated_text_examples(
        artifact=train_artifact,
        generated_rows=train_generated_rows,
        transformed_targets=transformed_train,
    )
    validation_examples = build_generated_text_examples(
        artifact=validation_artifact,
        generated_rows=validation_generated_rows,
        transformed_targets=transformed_validation,
    )
    collate_fn = make_collate_fn(ar_bundle.tokenizer, args.max_ar_length)
    train_loader = DataLoader(
        GeneratedTextARDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        GeneratedTextARDataset(validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in ar_bundle.model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    tqdm = import_tqdm()

    print_section(6, 10, "Adapting Qwen AR")
    metrics_rows = []
    best_fve = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_predictions: torch.Tensor | None = None
    best_metadata_rows: list[dict] | None = None
    output_files: dict[str, str] = {}
    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(
            model=ar_bundle.model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            tqdm=tqdm,
        )
        validation_metrics, predictions, metadata_rows = evaluate_adapted_ar(
            ar_bundle=ar_bundle,
            dataloader=validation_loader,
            device=device,
            original_validation_targets=validation_artifact.activations,
            seed=args.seed,
        )
        is_best = validation_metrics["validation_fve"] > best_fve
        if is_best:
            best_fve = validation_metrics["validation_fve"]
            best_epoch = epoch
            best_metrics = validation_metrics
            best_predictions = predictions
            best_metadata_rows = metadata_rows
            output_files = save_adapted_qwen_ar_checkpoint(
                output_dir=output_dir,
                ar_bundle=ar_bundle,
                target_transform=ar_bundle.target_transform,
                epoch=epoch,
                validation_metrics=validation_metrics,
                args=args,
            )
        metrics_rows.append(
            metric_row(
                epoch=epoch,
                train_mse=train_mse,
                validation_metrics=validation_metrics,
                is_best=is_best,
            )
        )
        print(
            f"epoch {epoch:03d}: train_mse={train_mse:.6f}, "
            f"validation_fve={validation_metrics['validation_fve']:.6f}, "
            f"validation_mse={validation_metrics['validation_mse']:.6f}"
        )

    if best_metrics is None or best_predictions is None or best_metadata_rows is None:
        raise ValueError("Adaptation finished without a best checkpoint.")

    print_section(7, 10, "Writing generated explanations")
    train_generated_path = output_dir / "train_generated_explanations.jsonl"
    validation_generated_path = output_dir / "validation_generated_explanations.jsonl"
    write_jsonl(train_generated_path, train_generated_rows)
    write_jsonl(validation_generated_path, validation_generated_rows)
    output_files.update(
        {
            "train_generated_explanations": train_generated_path.name,
            "validation_generated_explanations": validation_generated_path.name,
        }
    )

    print_section(8, 10, "Writing validation artifacts")
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

    print_section(9, 10, "Writing manifest")
    manifest_path = output_dir / "train_qwen_nla_reconstruction_manifest.json"
    manifest = build_manifest_payload(
        args=args,
        train_artifact=train_artifact,
        validation_artifact=validation_artifact,
        av_checkpoint_summary=checkpoint_summary(av_bundle),
        ar_checkpoint_summary=checkpoint_summary(ar_bundle),
        target_transform=ar_bundle.target_transform,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        output_files={**output_files, "manifest": manifest_path.name},
    )
    write_json(manifest_path, manifest)
    for filename in {**output_files, "manifest": manifest_path.name}.values():
        print(f"Wrote {output_dir / filename}")

    print_section(10, 10, "Result")
    print("SUCCESS: Phase 10c Qwen reconstruction-aware adaptation completed successfully.")


if __name__ == "__main__":
    main()
