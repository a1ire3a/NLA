"""Summarize completed AR training runs without loading tensors or models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = (
    "run_dir",
    "target_transform",
    "text_field",
    "freeze_text_model",
    "best_epoch",
    "validation_fve",
    "validation_mse",
    "validation_rmse",
    "validation_cosine_mean",
    "validation_train_mean_baseline_fve",
    "validation_train_mean_baseline_mse",
    "beats_validation_train_mean_baseline",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize AR manifests and training metrics across run directories."
    )
    parser.add_argument("run_dirs", nargs="+", help="AR checkpoint output directories.")
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def read_best_training_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    for row in rows:
        if str(row.get("is_best", "")).lower() == "true":
            return row
    return rows[-1]


def first_present(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "train_ar_manifest.json")
    best_metrics = manifest.get("best_validation_metrics", {})
    baseline = manifest.get("validation_train_mean_baseline", {})
    best_training_row = read_best_training_row(run_dir / "training_metrics.csv") or {}
    cli_args = manifest.get("cli_args", {})

    baseline_fve = baseline.get(
        "validation_train_mean_baseline_fve",
        best_training_row.get("validation_train_mean_baseline_fve"),
    )
    baseline_mse = baseline.get(
        "validation_train_mean_baseline_mse",
        best_training_row.get("validation_train_mean_baseline_mse"),
    )
    return {
        "run_dir": str(run_dir),
        "target_transform": first_present(
            manifest.get("target_transform"),
            cli_args.get("target_transform"),
            best_training_row.get("target_transform"),
        ),
        "text_field": first_present(manifest.get("text_field"), cli_args.get("text_field")),
        "freeze_text_model": first_present(
            manifest.get("freeze_text_model"),
            cli_args.get("freeze_text_model"),
        ),
        "best_epoch": first_present(manifest.get("best_epoch"), best_training_row.get("epoch")),
        "validation_fve": first_present(
            best_metrics.get("fve"),
            best_training_row.get("validation_fve"),
        ),
        "validation_mse": first_present(
            best_metrics.get("mse"),
            best_training_row.get("validation_mse"),
        ),
        "validation_rmse": first_present(
            best_metrics.get("rmse"),
            best_training_row.get("validation_rmse"),
        ),
        "validation_cosine_mean": first_present(
            best_metrics.get("cosine_mean"),
            best_training_row.get("validation_cosine_mean"),
        ),
        "validation_train_mean_baseline_fve": baseline_fve,
        "validation_train_mean_baseline_mse": baseline_mse,
        "beats_validation_train_mean_baseline": manifest.get(
            "beats_validation_train_mean_baseline"
        ),
    }


def write_rows(path: Path | None, rows: list[dict[str, Any]]) -> None:
    handle = path.open("w", encoding="utf-8", newline="") if path else None
    try:
        output = handle
        if output is None:
            import sys

            output = sys.stdout
        writer = csv.DictWriter(output, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})
    finally:
        if handle is not None:
            handle.close()


def main() -> None:
    args = parse_args()
    rows = [summarize_run(Path(run_dir)) for run_dir in args.run_dirs]
    output_csv = Path(args.output_csv) if args.output_csv else None
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output_csv, rows)


if __name__ == "__main__":
    main()
