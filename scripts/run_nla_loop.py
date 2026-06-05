"""Run the full NLA loop: activation -> AV text -> AR activation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

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
from nla_code_interp.models import ActivationVerbalizer, TextActivationReconstructor  # noqa: E402
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.run_evaluation import (  # noqa: E402
    ActivationArtifact,
    copy_manifest_summary,
    load_activation_artifact,
    prepare_output_dir,
)
from scripts.train_ar import (  # noqa: E402
    TargetTransform,
    ensure_tokenizer_padding as ensure_ar_tokenizer_padding,
    import_transformers as import_ar_tokenizer,
)
from scripts.train_av import (  # noqa: E402
    build_av_examples,
    ensure_tokenizer_padding as ensure_av_tokenizer_padding,
    import_transformers as import_av_tokenizer,
    parse_fallback_fields as parse_av_fallback_fields,
)


SCHEMA_VERSION = "phase8_full_nla_loop_v1"
METHODS = ("nla", "mean", "zero", "shuffled")
METRIC_COLUMNS = (
    "run_name",
    "method",
    "num_examples",
    "activation_dim",
    "fve",
    "mse",
    "rmse",
    "mean_l2_error",
    "cosine_mean",
    "cosine_std",
    "cosine_min",
    "cosine_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate activation -> AV explanation -> AR activation reconstruction."
    )
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--av_checkpoint_dir", required=True)
    parser.add_argument("--ar_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive when set, got {args.limit}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {args.max_new_tokens}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_paths(output_dir: Path, run_name: str) -> dict[str, Path]:
    return {
        "metrics_csv": output_dir / f"{run_name}_nla_metrics.csv",
        "metrics_json": output_dir / f"{run_name}_nla_metrics.json",
        "generated": output_dir / f"{run_name}_generated_explanations.jsonl",
        "per_example": output_dir / f"{run_name}_per_example_metrics.jsonl",
        "manifest": output_dir / f"{run_name}_manifest.json",
    }


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


def load_checkpoint(checkpoint_dir: Path, *, label: str, device: torch.device) -> dict:
    checkpoint_path = checkpoint_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing {label} checkpoint file: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict in {checkpoint_path}, got {type(checkpoint)}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is missing model_state_dict.")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{checkpoint_path} is missing config.")
    return checkpoint


def load_av_model_and_tokenizer(
    *,
    checkpoint_dir: Path,
    activation_dim: int,
    device: torch.device,
) -> tuple[ActivationVerbalizer, Any, dict, dict]:
    checkpoint = load_checkpoint(checkpoint_dir, label="AV", device=device)
    config = checkpoint["config"]
    checkpoint_activation_dim = int(config["activation_dim"])
    if checkpoint_activation_dim != activation_dim:
        raise ValueError(
            f"AV checkpoint activation_dim={checkpoint_activation_dim} does not match "
            f"target activation dim {activation_dim}."
        )

    AutoTokenizer = import_av_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(config["text_model_name_or_path"])
    ensure_av_tokenizer_padding(tokenizer)
    model = ActivationVerbalizer(
        text_model_name_or_path=config["text_model_name_or_path"],
        activation_dim=activation_dim,
        freeze_lm=bool(config.get("freeze_lm", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, tokenizer, config, checkpoint


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


def target_transform_from_checkpoint_state(state: dict[str, Any]) -> TargetTransform:
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
    return TargetTransform(name=name, mean=mean, std=std, eps=eps)


def load_ar_model_tokenizer_and_transform(
    *,
    checkpoint_dir: Path,
    activation_dim: int,
    device: torch.device,
) -> tuple[TextActivationReconstructor, Any, TargetTransform, dict, dict]:
    checkpoint = load_checkpoint(checkpoint_dir, label="AR", device=device)
    config = checkpoint["config"]
    checkpoint_activation_dim = int(config["activation_dim"])
    if checkpoint_activation_dim != activation_dim:
        raise ValueError(
            f"AR checkpoint activation_dim={checkpoint_activation_dim} does not match "
            f"target activation dim {activation_dim}."
        )
    transform_state = checkpoint.get("target_transform_state")
    if not isinstance(transform_state, dict):
        raise ValueError(f"{checkpoint_dir / 'model.pt'} is missing target_transform_state.")
    target_transform = target_transform_from_checkpoint_state(transform_state)

    AutoTokenizer = import_ar_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(config["text_model_name_or_path"])
    ensure_ar_tokenizer_padding(tokenizer)
    model = TextActivationReconstructor(
        text_model_name_or_path=config["text_model_name_or_path"],
        activation_dim=activation_dim,
        pooling=config.get("pooling", "mean"),
        projection_hidden_dim=config.get("projection_hidden_dim"),
        dropout=float(config.get("dropout", 0.0)),
        freeze_text_model=bool(config.get("freeze_text_model", True)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, tokenizer, target_transform, config, checkpoint


@torch.no_grad()
def generate_explanation_rows(
    *,
    model: ActivationVerbalizer,
    tokenizer,
    config: dict,
    artifact: ActivationArtifact,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    fallback_fields = parse_av_fallback_fields(config.get("fallback_text_fields", "code,prompt"))
    examples = build_av_examples(
        artifact.metadata_rows,
        target_text_field=config.get("target_text_field", "reference_description"),
        fallback_text_fields=fallback_fields,
    )

    rows = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_activations = artifact.activations[start : start + batch_size].to(device)
        generated_ids = model.greedy_generate(
            activations=batch_activations,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_texts = tokenizer.batch_decode(
            generated_ids.detach().cpu(),
            skip_special_tokens=True,
        )
        for example, generated_text in zip(batch_examples, generated_texts, strict=True):
            rows.append(
                generated_explanation_row(
                    source_index=example.source_index,
                    metadata=example.metadata,
                    target_text=example.target_text,
                    generated_text=generated_text,
                )
            )
    return rows


def generated_explanation_row(
    *,
    source_index: int,
    metadata: dict[str, Any],
    target_text: str,
    generated_text: str,
) -> dict[str, Any]:
    return {
        "activation_index": int(metadata.get("activation_index", source_index)),
        "example_id": metadata.get("example_id"),
        "target_text": target_text,
        "reference_description": metadata.get("reference_description"),
        "generated_text": generated_text.strip(),
        "split": metadata.get("split"),
        "language": metadata.get("language"),
        "transformation_type": metadata.get("transformation_type"),
    }


@torch.no_grad()
def reconstruct_with_ar(
    *,
    model: TextActivationReconstructor,
    tokenizer,
    target_transform: TargetTransform,
    generated_rows: list[dict[str, Any]],
    config: dict,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    max_length = int(config.get("max_length", 256))
    prediction_batches = []
    for start in range(0, len(generated_rows), batch_size):
        batch_rows = generated_rows[start : start + batch_size]
        texts = [row["generated_text"] for row in batch_rows]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
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
    transformed_predictions = torch.cat(prediction_batches, dim=0)
    return target_transform.inverse_transform(transformed_predictions).detach().cpu().float()


def build_reconstruction_methods(
    *,
    original: torch.Tensor,
    nla_reconstruction: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    return {
        "nla": nla_reconstruction,
        "mean": baseline_mean_reconstruction(original, original),
        "zero": baseline_zero_reconstruction(original),
        "shuffled": baseline_shuffled_reconstruction(original, seed),
    }


def metric_row(
    *,
    run_name: str,
    method: str,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, Any]:
    summary = summarize_reconstruction(original, reconstructed)
    return {
        "run_name": run_name,
        "method": method,
        "num_examples": original.shape[0],
        "activation_dim": original.shape[1],
        **summary,
    }


def metric_rows_for_methods(
    *,
    run_name: str,
    original: torch.Tensor,
    reconstructions: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        if method not in reconstructions:
            raise ValueError(f"Missing reconstruction method {method!r}")
        rows.append(
            metric_row(
                run_name=run_name,
                method=method,
                original=original,
                reconstructed=reconstructions[method],
            )
        )
    return rows


def per_example_nla_metric_rows(
    *,
    metadata_rows: list[dict],
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(metadata_rows) != original.shape[0] or len(generated_rows) != original.shape[0]:
        raise ValueError(
            "metadata, generated rows, and activation rows must have matching counts: "
            f"{len(metadata_rows)}, {len(generated_rows)}, {original.shape[0]}"
        )
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


def validate_reconstruction_shape(original: torch.Tensor, reconstructed: torch.Tensor) -> None:
    if reconstructed.shape != original.shape:
        raise ValueError(
            f"NLA reconstruction shape {tuple(reconstructed.shape)} does not match "
            f"target activations {tuple(original.shape)}."
        )


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in METRIC_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def checkpoint_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "schema_version": checkpoint.get("schema_version"),
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config", {}),
    }
    if "validation_loss" in checkpoint:
        summary["validation_loss"] = checkpoint["validation_loss"]
    if "validation_metrics" in checkpoint:
        summary["validation_metrics"] = checkpoint["validation_metrics"]
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    activation_dir = Path(args.activation_dir)
    av_checkpoint_dir = Path(args.av_checkpoint_dir)
    ar_checkpoint_dir = Path(args.ar_checkpoint_dir)
    output_dir = Path(args.output_dir)
    paths = output_paths(output_dir, args.run_name)
    device = resolve_device()

    print_section(1, 8, "Loading activation artifacts")
    full_artifact = load_activation_artifact(activation_dir)
    artifact = subset_artifact(full_artifact, args.limit)
    prepare_output_dir(output_dir, paths, args.overwrite)
    print(f"Target activations: {tuple(artifact.activations.shape)}")
    if args.limit is not None:
        print(f"Applied limit: {args.limit}")

    print_section(2, 8, "Loading AV checkpoint")
    av_model, av_tokenizer, av_config, av_checkpoint = load_av_model_and_tokenizer(
        checkpoint_dir=av_checkpoint_dir,
        activation_dim=artifact.activation_dim,
        device=device,
    )
    print(f"AV model: {av_config['text_model_name_or_path']}")
    print(f"Device: {device}")

    print_section(3, 8, "Generating AV explanations")
    generated_rows = generate_explanation_rows(
        model=av_model,
        tokenizer=av_tokenizer,
        config=av_config,
        artifact=artifact,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    if len(generated_rows) != artifact.num_examples:
        raise ValueError(
            f"Generated row count {len(generated_rows)} does not match "
            f"activation rows {artifact.num_examples}."
        )
    print(f"Generated explanations: {len(generated_rows)}")

    print_section(4, 8, "Loading AR checkpoint")
    ar_model, ar_tokenizer, target_transform, ar_config, ar_checkpoint = (
        load_ar_model_tokenizer_and_transform(
            checkpoint_dir=ar_checkpoint_dir,
            activation_dim=artifact.activation_dim,
            device=device,
        )
    )
    print(f"AR model: {ar_config['text_model_name_or_path']}")
    print(f"AR target transform: {target_transform.name}")

    print_section(5, 8, "Reconstructing activations with AR")
    nla_reconstruction = reconstruct_with_ar(
        model=ar_model,
        tokenizer=ar_tokenizer,
        target_transform=target_transform,
        generated_rows=generated_rows,
        config=ar_config,
        device=device,
        batch_size=args.batch_size,
    )
    validate_reconstruction_shape(artifact.activations, nla_reconstruction)
    print(f"NLA reconstruction: {tuple(nla_reconstruction.shape)}")

    print_section(6, 8, "Computing metrics")
    reconstructions = build_reconstruction_methods(
        original=artifact.activations,
        nla_reconstruction=nla_reconstruction,
        seed=args.seed,
    )
    metrics = metric_rows_for_methods(
        run_name=args.run_name,
        original=artifact.activations,
        reconstructions=reconstructions,
    )
    per_example_rows = per_example_nla_metric_rows(
        metadata_rows=artifact.metadata_rows,
        original=artifact.activations,
        reconstructed=nla_reconstruction,
        generated_rows=generated_rows,
    )
    for row in metrics:
        print(f"{row['method']}: FVE={row['fve']:.6f}, MSE={row['mse']:.6f}")

    print_section(7, 8, "Writing outputs")
    write_metrics_csv(paths["metrics_csv"], metrics)
    write_json(paths["metrics_json"], metrics)
    write_jsonl(paths["generated"], generated_rows)
    write_jsonl(paths["per_example"], per_example_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "activation_dir": str(activation_dir),
        "av_checkpoint_dir": str(av_checkpoint_dir),
        "ar_checkpoint_dir": str(ar_checkpoint_dir),
        "output_dir": str(output_dir),
        "run_name": args.run_name,
        "limit": args.limit,
        "num_examples": artifact.num_examples,
        "activation_dim": artifact.activation_dim,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "activation_artifact_manifest_summary": copy_manifest_summary(artifact.manifest),
        "av_checkpoint_summary": checkpoint_summary(av_checkpoint),
        "ar_checkpoint_summary": checkpoint_summary(ar_checkpoint),
        "target_transform": target_transform.name,
        "metrics": metrics,
        "output_files": {key: str(path) for key, path in paths.items()},
        "notes": [
            "AR input text is AV-generated natural language, even though the selected "
            "Phase 6c AR checkpoint was trained on code text.",
            "Mean baseline is computed from the same target subset in this Phase 8 CLI.",
        ],
    }
    write_json(paths["manifest"], manifest)
    for path in paths.values():
        print(f"Wrote {path}")

    print_section(8, 8, "Result")
    print("SUCCESS: Phase 8 full NLA loop completed successfully.")


if __name__ == "__main__":
    main()
