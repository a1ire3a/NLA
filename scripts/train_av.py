"""Train the Activation Verbalizer (AV): activation -> text."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.data import iter_jsonl, write_jsonl  # noqa: E402
from nla_code_interp.models import ActivationVerbalizer, count_trainable_parameters  # noqa: E402
from nla_code_interp.utils import set_seed  # noqa: E402


SCHEMA_VERSION = "phase7_av_training_v1"
TARGET_TEXT_FIELDS = ("reference_description", "code", "prompt")
METRIC_COLUMNS = ("epoch", "train_loss", "validation_loss", "is_best")


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


@dataclass(frozen=True)
class AVExample:
    source_index: int
    target_text: str
    selected_target_field: str
    metadata: dict[str, Any]


class AVDataset(Dataset):
    """Dataset for activation vectors paired with target text."""

    def __init__(
        self,
        *,
        examples: list[AVExample],
        activations: torch.Tensor,
    ) -> None:
        if len(examples) != activations.shape[0]:
            raise ValueError(
                f"Example count {len(examples)} does not match activation rows "
                f"{activations.shape[0]}"
            )
        self.examples = examples
        self.activations = activations

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "activation": self.activations[index],
            "target_text": example.target_text,
            "selected_target_field": example.selected_target_field,
            "metadata": example.metadata,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised activation verbalizer.")
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--validation_activation_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_model_name_or_path", default="distilgpt2")
    parser.add_argument(
        "--target_text_field",
        choices=TARGET_TEXT_FIELDS,
        default="reference_description",
    )
    parser.add_argument("--fallback_text_fields", default="code,prompt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument(
        "--freeze_lm",
        dest="freeze_lm",
        action="store_true",
        default=False,
        help="Freeze the causal LM and train only the activation projection.",
    )
    parser.add_argument(
        "--unfreeze_lm",
        dest="freeze_lm",
        action="store_false",
        help="Train the causal LM as well as the activation projection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")
    if args.max_target_length <= 0:
        raise ValueError(
            f"max_target_length must be positive, got {args.max_target_length}"
        )


def parse_fallback_fields(value: str) -> list[str]:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    invalid = [field for field in fields if field not in TARGET_TEXT_FIELDS]
    if invalid:
        raise ValueError(
            f"Invalid fallback target field(s): {invalid}. "
            f"Allowed fields: {TARGET_TEXT_FIELDS}"
        )
    return fields


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def select_target_text(
    row: dict,
    *,
    target_text_field: str,
    fallback_text_fields: list[str],
) -> tuple[str, str]:
    """Select target text with deterministic fallback when all fields are empty."""
    for field in ordered_unique([target_text_field, *fallback_text_fields]):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return deterministic_target_fallback(row), "deterministic_fallback"


def deterministic_target_fallback(row: dict) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    raw_task_id = metadata.get("raw_task_id")
    example_id = row.get("example_id") or raw_task_id or "unknown_example"
    language = row.get("language") or "unknown"
    transformation_type = row.get("transformation_type") or "unknown_transform"
    return (
        f"Describe the {language} function for example {example_id} "
        f"with {transformation_type} transformation."
    )


def build_av_examples(
    metadata_rows: list[dict],
    *,
    target_text_field: str,
    fallback_text_fields: list[str],
) -> list[AVExample]:
    examples = []
    for index, row in enumerate(metadata_rows):
        target_text, selected_field = select_target_text(
            row,
            target_text_field=target_text_field,
            fallback_text_fields=fallback_text_fields,
        )
        metadata = dict(row)
        metadata["av_target_field"] = selected_field
        metadata["av_target_text"] = target_text
        examples.append(
            AVExample(
                source_index=index,
                target_text=target_text,
                selected_target_field=selected_field,
                metadata=metadata,
            )
        )
    return examples


def target_field_counts(examples: list[AVExample]) -> dict[str, int]:
    return dict(sorted(Counter(example.selected_target_field for example in examples).items()))


def text_length_summary(examples: list[AVExample]) -> dict[str, float | int]:
    lengths = [len(example.target_text) for example in examples]
    if not lengths:
        raise ValueError("Cannot summarize text lengths for empty examples.")
    return {
        "count": len(lengths),
        "mean_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }


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
    if artifact.activations.ndim != 2:
        raise ValueError(
            f"Activation tensor must be 2D, got {artifact.activations.shape} "
            f"in {artifact.activation_dir}"
        )
    if artifact.activations.shape[0] == 0 or artifact.activations.shape[1] == 0:
        raise ValueError(f"Activation tensor must be non-empty, got {artifact.activations.shape}")
    if len(artifact.metadata_rows) != artifact.activations.shape[0]:
        raise ValueError(
            f"Metadata row count {len(artifact.metadata_rows)} does not match "
            f"activation rows {artifact.activations.shape[0]} in {artifact.activation_dir}"
        )
    manifest_examples = artifact.manifest.get("num_examples")
    if manifest_examples is not None and int(manifest_examples) != artifact.num_examples:
        raise ValueError(
            f"Manifest num_examples={manifest_examples} does not match "
            f"activation rows {artifact.num_examples} in {artifact.activation_dir}"
        )
    manifest_dim = artifact.manifest.get("activation_dim")
    if manifest_dim is not None and int(manifest_dim) != artifact.activation_dim:
        raise ValueError(
            f"Manifest activation_dim={manifest_dim} does not match "
            f"activation dim {artifact.activation_dim} in {artifact.activation_dir}"
        )


def validate_train_validation_artifacts(
    train_artifact: ActivationArtifact,
    validation_artifact: ActivationArtifact,
) -> None:
    if train_artifact.activation_dim != validation_artifact.activation_dim:
        raise ValueError(
            f"Validation activation dim {validation_artifact.activation_dim} does not "
            f"match train activation dim {train_artifact.activation_dim}."
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


def import_transformers():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformers. Install project dependencies with "
            "`pip install -r requirements.txt` inside the project environment."
        ) from exc
    return AutoTokenizer


def import_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        return lambda iterable, **_kwargs: iterable
    return tqdm


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no pad token and no eos token to reuse for padding.")
    tokenizer.pad_token = tokenizer.eos_token


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_collate_fn(tokenizer, max_target_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [append_eos(item["target_text"], tokenizer.eos_token) for item in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if "attention_mask" not in tokenized:
            tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"])
        return {
            "activations": torch.stack([item["activation"] for item in batch], dim=0),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "metadata": [item["metadata"] for item in batch],
            "target_texts": [item["target_text"] for item in batch],
        }

    return collate


def append_eos(text: str, eos_token: str | None) -> str:
    if not eos_token or text.endswith(eos_token):
        return text
    return text + eos_token


def move_batch_to_device(
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["activations"].to(device),
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
    )


def train_one_epoch(
    *,
    model: ActivationVerbalizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tqdm,
) -> float:
    model.train()
    if model.freeze_lm:
        model.language_model.eval()
    loss_total = 0.0
    example_count = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        activations, input_ids, attention_mask = move_batch_to_device(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            activations=activations,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        if outputs.loss is None:
            raise ValueError("Language model did not return a training loss.")
        outputs.loss.backward()
        optimizer.step()
        batch_size = activations.shape[0]
        loss_total += outputs.loss.item() * batch_size
        example_count += batch_size
    if example_count == 0:
        raise ValueError("Training dataloader produced no examples.")
    return loss_total / example_count


@torch.no_grad()
def evaluate_loss(
    *,
    model: ActivationVerbalizer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    loss_total = 0.0
    example_count = 0
    for batch in dataloader:
        activations, input_ids, attention_mask = move_batch_to_device(batch, device=device)
        outputs = model(
            activations=activations,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        if outputs.loss is None:
            raise ValueError("Language model did not return a validation loss.")
        batch_size = activations.shape[0]
        loss_total += outputs.loss.item() * batch_size
        example_count += batch_size
    if example_count == 0:
        raise ValueError("Validation dataloader produced no examples.")
    return loss_total / example_count


@torch.no_grad()
def generate_rows(
    *,
    model: ActivationVerbalizer,
    tokenizer,
    examples: list[AVExample],
    activations: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    model.eval()
    rows = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_activations = activations[start : start + batch_size].to(device)
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
            metadata = example.metadata
            rows.append(
                {
                    "activation_index": int(
                        metadata.get("activation_index", example.source_index)
                    ),
                    "example_id": metadata.get("example_id"),
                    "target_text": example.target_text,
                    "generated_text": generated_text.strip(),
                    "split": metadata.get("split"),
                    "language": metadata.get("language"),
                    "transformation_type": metadata.get("transformation_type"),
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
        "input_path",
        "model_name_or_path",
        "layer_index",
        "token_position",
        "num_examples",
        "activation_dim",
        "activation_shape",
        "activation_dtype",
        "truncation_count",
    )
    return {key: manifest[key] for key in keys if key in manifest}


def clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def checkpoint_payload(
    *,
    model: ActivationVerbalizer,
    args: argparse.Namespace,
    activation_dim: int,
    epoch: int,
    validation_loss: float,
    train_artifact: ActivationArtifact,
    validation_artifact: ActivationArtifact,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "config": {
            "text_model_name_or_path": args.text_model_name_or_path,
            "activation_dim": activation_dim,
            "freeze_lm": args.freeze_lm,
            "target_text_field": args.target_text_field,
            "fallback_text_fields": args.fallback_text_fields,
            "max_target_length": args.max_target_length,
        },
        "epoch": epoch,
        "validation_loss": validation_loss,
        "train_activation_artifact_manifest_summary": copy_manifest_summary(
            train_artifact.manifest
        ),
        "validation_activation_artifact_manifest_summary": copy_manifest_summary(
            validation_artifact.manifest
        ),
    }


def save_checkpoint(
    path: Path,
    *,
    model: ActivationVerbalizer,
    args: argparse.Namespace,
    activation_dim: int,
    epoch: int,
    validation_loss: float,
    train_artifact: ActivationArtifact,
    validation_artifact: ActivationArtifact,
) -> None:
    torch.save(
        checkpoint_payload(
            model=model,
            args=args,
            activation_dim=activation_dim,
            epoch=epoch,
            validation_loss=validation_loss,
            train_artifact=train_artifact,
            validation_artifact=validation_artifact,
        ),
        path,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    activation_dir = Path(args.activation_dir)
    validation_activation_dir = Path(args.validation_activation_dir)
    output_dir = Path(args.output_dir)
    fallback_fields = parse_fallback_fields(args.fallback_text_fields)
    device = resolve_device()

    print_section(1, 8, "Loading activation artifacts")
    train_artifact = load_activation_artifact(activation_dir)
    validation_artifact = load_activation_artifact(validation_activation_dir)
    validate_train_validation_artifacts(train_artifact, validation_artifact)
    prepare_output_dir(output_dir, args.overwrite)
    print(f"Train activations: {tuple(train_artifact.activations.shape)}")
    print(f"Validation activations: {tuple(validation_artifact.activations.shape)}")

    print_section(2, 8, "Building AV datasets")
    train_examples = build_av_examples(
        train_artifact.metadata_rows,
        target_text_field=args.target_text_field,
        fallback_text_fields=fallback_fields,
    )
    validation_examples = build_av_examples(
        validation_artifact.metadata_rows,
        target_text_field=args.target_text_field,
        fallback_text_fields=fallback_fields,
    )
    print(f"Target text field: {args.target_text_field}")
    print(f"Fallback fields: {fallback_fields}")
    print(f"Train target field counts: {target_field_counts(train_examples)}")
    print(f"Validation target field counts: {target_field_counts(validation_examples)}")
    print(f"Train target text lengths: {text_length_summary(train_examples)}")
    print(f"Validation target text lengths: {text_length_summary(validation_examples)}")

    print_section(3, 8, "Loading tokenizer and AV model")
    AutoTokenizer = import_transformers()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name_or_path)
        ensure_tokenizer_padding(tokenizer)
        model = ActivationVerbalizer(
            text_model_name_or_path=args.text_model_name_or_path,
            activation_dim=train_artifact.activation_dim,
            freeze_lm=args.freeze_lm,
        )
    except Exception as exc:
        print("ERROR: tokenizer/model loading failed.")
        print(f"Exception: {type(exc).__name__}: {exc}")
        raise
    model = model.to(device)
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"LM frozen: {args.freeze_lm}")
    print(f"LM embedding dim: {model.lm_embedding_dim}")
    print(f"Device: {device}")

    print_section(4, 8, "Training setup")
    train_dataset = AVDataset(
        examples=train_examples,
        activations=train_artifact.activations,
    )
    validation_dataset = AVDataset(
        examples=validation_examples,
        activations=validation_artifact.activations,
    )
    collate_fn = make_collate_fn(tokenizer, args.max_target_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    trainable_parameters = count_trainable_parameters(model)
    if trainable_parameters == 0:
        raise ValueError("AV model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    print(f"Trainable parameters: {trainable_parameters}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    print_section(5, 8, "Training")
    tqdm = import_tqdm()
    metrics_rows = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            tqdm=tqdm,
        )
        validation_loss = evaluate_loss(
            model=model,
            dataloader=validation_loader,
            device=device,
        )
        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state_dict = clone_state_dict_to_cpu(model)
            save_checkpoint(
                output_dir / "model.pt",
                model=model,
                args=args,
                activation_dim=train_artifact.activation_dim,
                epoch=epoch,
                validation_loss=validation_loss,
                train_artifact=train_artifact,
                validation_artifact=validation_artifact,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "is_best": is_best,
        }
        metrics_rows.append(row)
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f}, "
            f"validation_loss={validation_loss:.6f}"
        )

    if best_state_dict is None:
        raise ValueError("Training finished without a best checkpoint.")
    model.load_state_dict(best_state_dict)

    print_section(6, 8, "Generating validation examples")
    generation_rows = generate_rows(
        model=model,
        tokenizer=tokenizer,
        examples=validation_examples,
        activations=validation_artifact.activations,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_target_length,
    )
    print(f"Generated validation rows: {len(generation_rows)}")

    print_section(7, 8, "Writing outputs")
    metrics_path = output_dir / "training_metrics.csv"
    generations_path = output_dir / "validation_generations.jsonl"
    manifest_path = output_dir / "train_av_manifest.json"
    write_metrics_csv(metrics_path, metrics_rows)
    write_jsonl(generations_path, generation_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "activation_dir": str(activation_dir),
        "validation_activation_dir": str(validation_activation_dir),
        "output_dir": str(output_dir),
        "text_model_name_or_path": args.text_model_name_or_path,
        "target_text_field": args.target_text_field,
        "fallback_text_fields": fallback_fields,
        "freeze_lm": args.freeze_lm,
        "max_target_length": args.max_target_length,
        "train_count": train_artifact.num_examples,
        "validation_count": validation_artifact.num_examples,
        "activation_dim": train_artifact.activation_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "train_target_field_counts": target_field_counts(train_examples),
        "validation_target_field_counts": target_field_counts(validation_examples),
        "train_text_length_summary": text_length_summary(train_examples),
        "validation_text_length_summary": text_length_summary(validation_examples),
        "train_activation_artifact_manifest_summary": copy_manifest_summary(
            train_artifact.manifest
        ),
        "validation_activation_artifact_manifest_summary": copy_manifest_summary(
            validation_artifact.manifest
        ),
        "output_files": {
            "model": "model.pt",
            "training_metrics": metrics_path.name,
            "validation_generations": generations_path.name,
            "manifest": manifest_path.name,
        },
    }
    write_json(manifest_path, manifest)
    for path in (output_dir / "model.pt", metrics_path, generations_path, manifest_path):
        print(f"Wrote {path}")

    print_section(8, 8, "Result")
    print("SUCCESS: Phase 7 AV training completed successfully.")


if __name__ == "__main__":
    main()
