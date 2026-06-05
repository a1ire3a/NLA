"""Run baseline evaluation over saved activation artifacts."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.data import iter_jsonl, write_jsonl  # noqa: E402
from nla_code_interp.metrics import (  # noqa: E402
    baseline_mean_reconstruction,
    baseline_shuffled_reconstruction,
    baseline_zero_reconstruction,
    per_example_cosine_similarity,
    per_example_l2_error,
    per_example_squared_error,
    summarize_reconstruction,
)


SCHEMA_VERSION = "phase4_baseline_eval_v1"
BASELINES = ("mean", "zero", "shuffled")
METRIC_COLUMNS = (
    "run_name",
    "baseline",
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


@dataclass(frozen=True)
class ActivationArtifact:
    activation_dir: Path
    activations: torch.Tensor
    metadata_rows: list[dict]
    manifest: dict[str, Any]

    @property
    def num_examples(self) -> int:
        return self.activations.shape[0]

    @property
    def activation_dim(self) -> int:
        return self.activations.shape[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trivial baselines for activation artifacts."
    )
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--reference_activation_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/reports/baselines")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return Path(args.activation_dir).name


def output_paths(output_dir: Path, run_name: str) -> dict[str, Path]:
    return {
        "csv": output_dir / f"{run_name}_baseline_metrics.csv",
        "json": output_dir / f"{run_name}_baseline_metrics.json",
        "per_example": output_dir / f"{run_name}_per_example_errors.jsonl",
        "manifest": output_dir / f"{run_name}_evaluation_manifest.json",
    }


def prepare_output_dir(output_dir: Path, paths: dict[str, Path], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output file(s) already exist: {joined}. Pass --overwrite to replace."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def load_activation_artifact(activation_dir: Path) -> ActivationArtifact:
    tensor_path = activation_dir / "activations.pt"
    metadata_path = activation_dir / "metadata.jsonl"
    manifest_path = activation_dir / "manifest.json"
    for path in (tensor_path, metadata_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required activation artifact file: {path}")

    activations = torch.load(tensor_path, map_location="cpu")
    if not isinstance(activations, torch.Tensor):
        raise TypeError(f"Expected tensor in {tensor_path}, got {type(activations)}")
    activations = activations.detach().cpu().float()
    metadata_rows = list(iter_jsonl(metadata_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifact = ActivationArtifact(
        activation_dir=activation_dir,
        activations=activations,
        metadata_rows=metadata_rows,
        manifest=manifest,
    )
    validate_activation_artifact(artifact)
    return artifact


def validate_activation_artifact(artifact: ActivationArtifact) -> None:
    activations = artifact.activations
    metadata_rows = artifact.metadata_rows
    manifest = artifact.manifest

    if activations.ndim != 2:
        raise ValueError(
            f"Activation tensor must be 2D, got {activations.shape} in {artifact.activation_dir}"
        )
    if activations.shape[0] == 0 or activations.shape[1] == 0:
        raise ValueError(f"Activation tensor must be non-empty, got {activations.shape}")
    if len(metadata_rows) != activations.shape[0]:
        raise ValueError(
            f"Metadata row count {len(metadata_rows)} does not match activation rows "
            f"{activations.shape[0]} in {artifact.activation_dir}"
        )

    manifest_examples = manifest.get("num_examples")
    if manifest_examples is not None and int(manifest_examples) != activations.shape[0]:
        raise ValueError(
            f"Manifest num_examples={manifest_examples} does not match activation rows "
            f"{activations.shape[0]} in {artifact.activation_dir}"
        )
    manifest_dim = manifest.get("activation_dim")
    if manifest_dim is not None and int(manifest_dim) != activations.shape[1]:
        raise ValueError(
            f"Manifest activation_dim={manifest_dim} does not match activation dim "
            f"{activations.shape[1]} in {artifact.activation_dir}"
        )

    indices = [row.get("activation_index") for row in metadata_rows]
    if all(index is not None for index in indices):
        expected = list(range(len(indices)))
        observed = [int(index) for index in indices]
        if observed != expected:
            raise ValueError(
                f"metadata activation_index values are not sequential in {artifact.activation_dir}"
            )


def build_baselines(
    *,
    target: torch.Tensor,
    reference: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    return {
        "mean": baseline_mean_reconstruction(reference, target),
        "zero": baseline_zero_reconstruction(target),
        "shuffled": baseline_shuffled_reconstruction(target, seed),
    }


def metric_row(
    *,
    run_name: str,
    baseline_name: str,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, Any]:
    summary = summarize_reconstruction(original, reconstructed)
    return {
        "run_name": run_name,
        "baseline": baseline_name,
        "num_examples": original.shape[0],
        "activation_dim": original.shape[1],
        **summary,
    }


def per_example_rows(
    *,
    metadata_rows: list[dict],
    baseline_name: str,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> list[dict]:
    squared_errors = per_example_squared_error(original, reconstructed)
    l2_errors = per_example_l2_error(original, reconstructed)
    cosine_similarities = per_example_cosine_similarity(original, reconstructed)

    rows = []
    for index, metadata in enumerate(metadata_rows):
        rows.append(
            {
                "activation_index": int(metadata.get("activation_index", index)),
                "example_id": metadata.get("example_id"),
                "split": metadata.get("split"),
                "language": metadata.get("language"),
                "transformation_type": metadata.get("transformation_type"),
                "baseline": baseline_name,
                "squared_error": squared_errors[index].item(),
                "l2_error": l2_errors[index].item(),
                "cosine_similarity": cosine_similarities[index].item(),
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


def copy_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "model_name_or_path",
        "layer_index",
        "token_position",
        "num_examples",
        "activation_dim",
        "activation_shape",
        "activation_dtype",
        "truncation_count",
        "activation_summary",
    )
    return {key: manifest[key] for key in keys if key in manifest}


def main() -> None:
    args = parse_args()
    activation_dir = Path(args.activation_dir)
    reference_dir = Path(args.reference_activation_dir) if args.reference_activation_dir else None
    output_dir = Path(args.output_dir)
    run_name = resolve_run_name(args)
    paths = output_paths(output_dir, run_name)

    print_section(1, 6, "Loading activation artifacts")
    target_artifact = load_activation_artifact(activation_dir)
    reference_artifact = (
        load_activation_artifact(reference_dir) if reference_dir else target_artifact
    )
    print(f"Target activations: {tuple(target_artifact.activations.shape)}")
    if reference_dir:
        print(f"Reference activations: {tuple(reference_artifact.activations.shape)}")

    print_section(2, 6, "Validating artifacts")
    if reference_artifact.activation_dim != target_artifact.activation_dim:
        raise ValueError(
            f"Reference activation dim {reference_artifact.activation_dim} does not match "
            f"target dim {target_artifact.activation_dim}."
        )
    prepare_output_dir(output_dir, paths, args.overwrite)
    print("Artifact validation passed.")

    print_section(3, 6, "Building baselines")
    baselines = build_baselines(
        target=target_artifact.activations,
        reference=reference_artifact.activations,
        seed=args.seed,
    )
    print(f"Baselines: {', '.join(baselines)}")

    print_section(4, 6, "Computing metrics")
    metric_rows = []
    per_example = []
    for baseline_name, reconstructed in baselines.items():
        row = metric_row(
            run_name=run_name,
            baseline_name=baseline_name,
            original=target_artifact.activations,
            reconstructed=reconstructed,
        )
        metric_rows.append(row)
        per_example.extend(
            per_example_rows(
                metadata_rows=target_artifact.metadata_rows,
                baseline_name=baseline_name,
                original=target_artifact.activations,
                reconstructed=reconstructed,
            )
        )
        print(f"{baseline_name}: FVE={row['fve']:.6f}, MSE={row['mse']:.6f}")

    print_section(5, 6, "Writing outputs")
    write_metrics_csv(paths["csv"], metric_rows)
    write_json(paths["json"], metric_rows)
    write_jsonl(paths["per_example"], per_example)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "activation_dir": str(activation_dir),
        "reference_activation_dir": str(reference_dir) if reference_dir else None,
        "run_name": run_name,
        "num_examples": target_artifact.num_examples,
        "activation_dim": target_artifact.activation_dim,
        "activation_artifact_manifest_summary": copy_manifest_summary(target_artifact.manifest),
        "reference_artifact_manifest_summary": (
            copy_manifest_summary(reference_artifact.manifest) if reference_dir else None
        ),
        "baseline_names": list(BASELINES),
        "output_files": {key: str(path) for key, path in paths.items()},
        "notes": [
            "When the mean baseline is computed from the same target tensor, FVE should be "
            "approximately 0 by construction.",
        ],
    }
    write_json(paths["manifest"], manifest)
    for path in paths.values():
        print(f"Wrote {path}")

    print_section(6, 6, "Result")
    print("SUCCESS: Phase 4 baseline evaluation completed successfully.")


if __name__ == "__main__":
    main()
