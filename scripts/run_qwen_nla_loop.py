"""Run the Qwen NLA loop: activation -> Qwen AV text -> Qwen AR activation."""

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
from nla_code_interp.qwen_models import (  # noqa: E402
    QwenARCheckpointBundle,
    QwenAVCheckpointBundle,
    load_qwen_ar_checkpoint,
    load_qwen_av_checkpoint,
)
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.train_ar import ActivationArtifact, load_activation_artifact  # noqa: E402
from scripts.train_qwen_av import (  # noqa: E402
    build_examples as build_qwen_av_examples,
    generate_rows as generate_qwen_av_rows,
    parse_fallback_fields as parse_qwen_av_fallback_fields,
)


SCHEMA_VERSION = "phase10c_qwen_nla_loop_v1"
METHODS = ("qwen_nla", "mean", "zero", "shuffled")
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
    parser = argparse.ArgumentParser(description="Run the Qwen full NLA loop.")
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--qwen_av_checkpoint_dir", required=True)
    parser.add_argument("--qwen_ar_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive when set, got {args.limit}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {args.max_new_tokens}")


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_paths(output_dir: Path, run_name: str) -> dict[str, Path]:
    return {
        "metrics_csv": output_dir / f"{run_name}_qwen_nla_metrics.csv",
        "metrics_json": output_dir / f"{run_name}_qwen_nla_metrics.json",
        "generated": output_dir / f"{run_name}_generated_explanations.jsonl",
        "per_example": output_dir / f"{run_name}_per_example_metrics.jsonl",
        "manifest": output_dir / f"{run_name}_manifest.json",
    }


def prepare_output_paths(paths: dict[str, Path], *, overwrite: bool) -> None:
    output_dirs = {path.parent for path in paths.values()}
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output file(s) already exist: {joined}. Pass --overwrite.")
    if overwrite:
        for path in existing:
            path.unlink()


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


def validate_checkpoint_dims(
    *,
    artifact: ActivationArtifact,
    av_bundle: QwenAVCheckpointBundle,
    ar_bundle: QwenARCheckpointBundle,
) -> None:
    av_dim = int(av_bundle.config["activation_dim"])
    ar_dim = int(ar_bundle.config["activation_dim"])
    if av_dim != artifact.activation_dim:
        raise ValueError(
            f"Qwen AV checkpoint activation_dim={av_dim} does not match "
            f"artifact activation_dim={artifact.activation_dim}."
        )
    if ar_dim != artifact.activation_dim:
        raise ValueError(
            f"Qwen AR checkpoint activation_dim={ar_dim} does not match "
            f"artifact activation_dim={artifact.activation_dim}."
        )


@torch.no_grad()
def generate_qwen_explanation_rows(
    *,
    artifact: ActivationArtifact,
    av_bundle: QwenAVCheckpointBundle,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    fallback_fields = parse_qwen_av_fallback_fields(
        av_bundle.config.get("fallback_text_fields", "prompt,code")
    )
    examples = build_qwen_av_examples(
        artifact=artifact,
        target_text_field=av_bundle.config.get("target_text_field", "reference_description"),
        fallback_text_fields=fallback_fields,
    )
    return generate_qwen_av_rows(
        model=av_bundle.model,
        tokenizer=av_bundle.tokenizer,
        examples=examples,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )


@torch.no_grad()
def reconstruct_with_qwen_ar(
    *,
    ar_bundle: QwenARCheckpointBundle,
    generated_rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int | None = None,
) -> torch.Tensor:
    model = ar_bundle.model
    tokenizer = ar_bundle.tokenizer
    model.eval()
    ar_max_length = int(max_length or ar_bundle.config.get("max_length", 256))
    prediction_batches = []
    for start in range(0, len(generated_rows), batch_size):
        batch_rows = generated_rows[start : start + batch_size]
        tokenized = tokenizer(
            [row["generated_text"] for row in batch_rows],
            padding=True,
            truncation=True,
            max_length=ar_max_length,
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
        raise ValueError("No generated rows were available for Qwen AR reconstruction.")
    transformed_predictions = torch.cat(prediction_batches, dim=0)
    return ar_bundle.target_transform.inverse_transform(transformed_predictions)


def build_reconstruction_methods(
    *,
    original: torch.Tensor,
    qwen_nla_reconstruction: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    return {
        "qwen_nla": qwen_nla_reconstruction,
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


def per_example_metric_rows(
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


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in METRIC_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def checkpoint_summary(bundle: QwenAVCheckpointBundle | QwenARCheckpointBundle) -> dict[str, Any]:
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

    activation_dir = Path(args.activation_dir)
    av_checkpoint_dir = Path(args.qwen_av_checkpoint_dir)
    ar_checkpoint_dir = Path(args.qwen_ar_checkpoint_dir)
    output_dir = Path(args.output_dir)
    paths = output_paths(output_dir, args.run_name)

    print_section(1, 8, "Loading activation artifact")
    artifact = subset_artifact(load_activation_artifact(activation_dir), args.limit)
    prepare_output_paths(paths, overwrite=args.overwrite)
    print(f"Activations: {tuple(artifact.activations.shape)}")

    print_section(2, 8, "Loading Qwen AV checkpoint")
    av_bundle = load_qwen_av_checkpoint(
        checkpoint_dir=av_checkpoint_dir,
        device=device,
        adapter_trainable=False,
    )
    print(f"Qwen AV model: {av_bundle.config['model_name_or_path']}")

    print_section(3, 8, "Generating explanations")
    generated_rows = generate_qwen_explanation_rows(
        artifact=artifact,
        av_bundle=av_bundle,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Generated explanations: {len(generated_rows)}")

    print_section(4, 8, "Loading Qwen AR checkpoint")
    ar_bundle = load_qwen_ar_checkpoint(
        checkpoint_dir=ar_checkpoint_dir,
        device=device,
        adapter_trainable=False,
    )
    validate_checkpoint_dims(artifact=artifact, av_bundle=av_bundle, ar_bundle=ar_bundle)
    print(f"Qwen AR model: {ar_bundle.config['model_name_or_path']}")
    print(f"Qwen AR target transform: {ar_bundle.target_transform.name}")

    print_section(5, 8, "Reconstructing activations")
    qwen_nla_reconstruction = reconstruct_with_qwen_ar(
        ar_bundle=ar_bundle,
        generated_rows=generated_rows,
        device=device,
        batch_size=args.batch_size,
    )
    if qwen_nla_reconstruction.shape != artifact.activations.shape:
        raise ValueError(
            f"Qwen NLA reconstruction shape {tuple(qwen_nla_reconstruction.shape)} "
            f"does not match activations {tuple(artifact.activations.shape)}."
        )
    print(f"Reconstruction: {tuple(qwen_nla_reconstruction.shape)}")

    print_section(6, 8, "Computing metrics")
    reconstructions = build_reconstruction_methods(
        original=artifact.activations,
        qwen_nla_reconstruction=qwen_nla_reconstruction,
        seed=args.seed,
    )
    metric_rows = metric_rows_for_methods(
        run_name=args.run_name,
        original=artifact.activations,
        reconstructions=reconstructions,
    )
    per_example_rows = per_example_metric_rows(
        metadata_rows=artifact.metadata_rows,
        original=artifact.activations,
        reconstructed=qwen_nla_reconstruction,
        generated_rows=generated_rows,
    )
    for row in metric_rows:
        print(f"{row['method']}: FVE={row['fve']:.6f}, MSE={row['mse']:.6f}")

    print_section(7, 8, "Writing outputs")
    write_metrics_csv(paths["metrics_csv"], metric_rows)
    write_json(paths["metrics_json"], metric_rows)
    write_jsonl(paths["generated"], generated_rows)
    write_jsonl(paths["per_example"], per_example_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "activation_dir": str(activation_dir),
        "qwen_av_checkpoint_dir": str(av_checkpoint_dir),
        "qwen_ar_checkpoint_dir": str(ar_checkpoint_dir),
        "output_dir": str(output_dir),
        "run_name": args.run_name,
        "limit": args.limit,
        "num_examples": artifact.num_examples,
        "activation_dim": artifact.activation_dim,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "qwen_av_checkpoint_summary": checkpoint_summary(av_bundle),
        "qwen_ar_checkpoint_summary": checkpoint_summary(ar_bundle),
        "metrics": metric_rows,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    write_json(paths["manifest"], manifest)
    for path in paths.values():
        print(f"Wrote {path}")

    print_section(8, 8, "Result")
    print("SUCCESS: Phase 10c Qwen NLA loop completed successfully.")


if __name__ == "__main__":
    main()
