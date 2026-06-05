"""Train a text-to-activation Activation Reconstructor baseline."""

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
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.data import iter_jsonl, write_jsonl  # noqa: E402
from nla_code_interp.metrics import (  # noqa: E402
    baseline_mean_reconstruction,
    per_example_cosine_similarity,
    per_example_l2_error,
    per_example_squared_error,
    summarize_reconstruction,
)
from nla_code_interp.models import (  # noqa: E402
    TextActivationReconstructor,
    count_trainable_parameters,
)
from nla_code_interp.utils import set_seed  # noqa: E402


SCHEMA_VERSION = "phase5b_ar_training_v2"
TEXT_FIELDS = ("reference_description", "prompt", "code")
TARGET_TRANSFORMS = ("raw", "center", "standardize")
METRIC_COLUMNS = (
    "epoch",
    "train_mse_loss",
    "target_transform",
    "validation_fve",
    "validation_mse",
    "validation_rmse",
    "validation_mean_l2_error",
    "validation_cosine_mean",
    "validation_cosine_std",
    "validation_cosine_min",
    "validation_cosine_max",
    "validation_prediction_norm_mean",
    "validation_target_norm_mean",
    "validation_original_norm_mean",
    "transformed_validation_mse",
    "validation_train_mean_baseline_fve",
    "validation_train_mean_baseline_mse",
    "is_best",
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


@dataclass(frozen=True)
class TextExample:
    source_index: int
    text: str
    selected_text_field: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TargetTransform:
    """Train-space target transform fit from training activations only."""

    name: str
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    eps: float = 1e-6

    @classmethod
    def fit(
        cls,
        name: str,
        train_targets: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> "TargetTransform":
        if name not in TARGET_TRANSFORMS:
            raise ValueError(f"Unsupported target transform {name!r}")
        train_targets = validate_target_matrix(train_targets, "train_targets")
        if name == "raw":
            return cls(name=name, eps=eps)

        mean = train_targets.mean(dim=0, keepdim=True)
        if name == "center":
            return cls(name=name, mean=mean, eps=eps)

        std = train_targets.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        return cls(name=name, mean=mean, std=std, eps=eps)

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        targets = validate_target_matrix(targets, "targets")
        if self.name == "raw":
            return targets.clone()
        if self.mean is None:
            raise ValueError(f"TargetTransform {self.name!r} is missing mean.")
        centered = targets - self.mean.to(device=targets.device, dtype=targets.dtype)
        if self.name == "center":
            return centered
        if self.std is None:
            raise ValueError("standardize TargetTransform is missing std.")
        return centered / self.std.to(device=targets.device, dtype=targets.dtype)

    def inverse_transform(self, predictions: torch.Tensor) -> torch.Tensor:
        predictions = validate_target_matrix(predictions, "predictions")
        if self.name == "raw":
            return predictions.clone()
        if self.mean is None:
            raise ValueError(f"TargetTransform {self.name!r} is missing mean.")
        mean = self.mean.to(device=predictions.device, dtype=predictions.dtype)
        if self.name == "center":
            return predictions + mean
        if self.std is None:
            raise ValueError("standardize TargetTransform is missing std.")
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
            state.update(
                {
                    "mean_shape": list(mean.shape),
                    "mean_norm": mean.norm(p=2).item(),
                    "mean": mean.squeeze(0).tolist(),
                }
            )
        if self.std is not None:
            std = self.std.detach().cpu().float()
            state.update(
                {
                    "std_shape": list(std.shape),
                    "std_mean": std.mean().item(),
                    "std_min": std.min().item(),
                    "std_max": std.max().item(),
                    "std": std.squeeze(0).tolist(),
                }
            )
        return state


@dataclass(frozen=True)
class ValidationResult:
    metrics: dict[str, float]
    transformed_predictions: torch.Tensor
    transformed_targets: torch.Tensor
    original_predictions: torch.Tensor
    original_targets: torch.Tensor
    metadata_rows: list[dict]


class ARDataset(Dataset):
    """Torch dataset for selected text inputs and activation targets."""

    def __init__(
        self,
        *,
        examples: list[TextExample],
        activations: torch.Tensor,
        indices: list[int],
    ) -> None:
        self.examples = examples
        self.activations = activations
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        source_index = self.indices[position]
        example = self.examples[source_index]
        return {
            "text": example.text,
            "target": self.activations[source_index],
            "metadata": example.metadata,
            "source_index": source_index,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight text-to-activation AR baseline."
    )
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--text_model_name_or_path",
        default="distilbert-base-uncased",
        help="Hugging Face text encoder name or local path.",
    )
    parser.add_argument(
        "--text_field",
        choices=TEXT_FIELDS,
        default="reference_description",
    )
    parser.add_argument("--fallback_text_fields", default="prompt,code")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument(
        "--freeze_text_model",
        dest="freeze_text_model",
        action="store_true",
        default=True,
        help="Freeze the text encoder and train only the projection head.",
    )
    parser.add_argument(
        "--unfreeze_text_model",
        dest="freeze_text_model",
        action="store_false",
        help="Make the text encoder trainable.",
    )
    parser.add_argument("--projection_hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--target_transform",
        choices=TARGET_TRANSFORMS,
        default=None,
        help="Target-space transform fit from train activations only.",
    )
    parser.add_argument(
        "--predict_residual_from_mean",
        action="store_true",
        help="Alias for --target_transform center.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--baseline_metrics_json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def parse_fallback_fields(value: str) -> list[str]:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    invalid = [field for field in fields if field not in TEXT_FIELDS]
    if invalid:
        raise ValueError(
            f"Invalid fallback text field(s): {invalid}. Allowed fields: {TEXT_FIELDS}"
        )
    return fields


def resolve_target_transform_arg(args: argparse.Namespace) -> str:
    requested = args.target_transform
    if args.predict_residual_from_mean:
        if requested in {"raw", "standardize"}:
            raise ValueError(
                "--predict_residual_from_mean is an alias for --target_transform center; "
                f"got conflicting --target_transform {requested!r}."
            )
        resolved = "center"
    else:
        resolved = requested or "raw"
    args.target_transform = resolved
    return resolved


def validate_target_matrix(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [num_examples, activation_dim]")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {tensor.shape}")
    return tensor.detach().float()


def select_text_for_row(
    row: dict,
    *,
    text_field: str,
    fallback_text_fields: list[str],
) -> tuple[str, str]:
    """Select the primary text field, falling back to configured alternatives."""
    for candidate in [text_field, *fallback_text_fields]:
        value = row.get(candidate)
        if isinstance(value, str) and value.strip():
            return value.strip(), candidate
    example_id = row.get("example_id", "<unknown>")
    fields = ", ".join([text_field, *fallback_text_fields])
    raise ValueError(f"Example {example_id!r} has no usable text in fields: {fields}")


def build_text_examples(
    metadata_rows: list[dict],
    *,
    text_field: str,
    fallback_text_fields: list[str],
) -> list[TextExample]:
    examples = []
    for index, row in enumerate(metadata_rows):
        text, selected_field = select_text_for_row(
            row,
            text_field=text_field,
            fallback_text_fields=fallback_text_fields,
        )
        metadata = dict(row)
        metadata.update(
            {
                "ar_source_index": index,
                "ar_text_field": selected_field,
                "ar_text": text,
            }
        )
        examples.append(
            TextExample(
                source_index=index,
                text=text,
                selected_text_field=selected_field,
                metadata=metadata,
            )
        )
    return examples


def text_field_counts(examples: list[TextExample]) -> dict[str, int]:
    return dict(sorted(Counter(example.selected_text_field for example in examples).items()))


def text_length_summary(examples: list[TextExample]) -> dict[str, float | int]:
    lengths = [len(example.text) for example in examples]
    if not lengths:
        raise ValueError("Cannot summarize text lengths for an empty example list.")
    return {
        "count": len(lengths),
        "mean_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }


def split_train_validation_indices(
    metadata_rows: list[dict],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], str]:
    """Use explicit train/validation splits when available, otherwise random split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in (0, 1), got {validation_fraction}"
        )
    num_examples = len(metadata_rows)
    if num_examples < 2:
        raise ValueError("At least two examples are required for train/validation split.")

    train_indices = []
    for index, row in enumerate(metadata_rows):
        if str(row.get("split", "")).lower() == "train":
            train_indices.append(index)
    validation_indices = [
        index
        for index, row in enumerate(metadata_rows)
        if str(row.get("split", "")).lower() in {"validation", "valid", "val"}
    ]
    if train_indices and validation_indices:
        return sorted(train_indices), sorted(validation_indices), "metadata_split"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_examples, generator=generator).tolist()
    validation_count = int(round(num_examples * validation_fraction))
    validation_count = max(1, min(num_examples - 1, validation_count))
    validation_set = set(permutation[:validation_count])
    train = [index for index in range(num_examples) if index not in validation_set]
    validation = [index for index in range(num_examples) if index in validation_set]
    return train, validation, "deterministic_random_split"


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
            f"Activation tensor must be 2D, got {activations.shape} in "
            f"{artifact.activation_dir}"
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
                f"metadata activation_index values are not sequential in "
                f"{artifact.activation_dir}"
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


def dtype_from_arg(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda, but CUDA is unavailable. Run on the CUDA server "
            "or pass --device cpu for a tiny local smoke check."
        )
    return device


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no pad token and no eos token to reuse for padding.")
    tokenizer.pad_token = tokenizer.eos_token


def print_model_loading_error(exc: Exception) -> None:
    print("ERROR: text tokenizer/model loading failed.")
    print(f"Exception: {type(exc).__name__}: {exc}")
    print("Likely causes:")
    print("- text model is not downloaded and network access is unavailable")
    print("- wrong --text_model_name_or_path")
    print("- missing Hugging Face access or authentication")
    print("- incompatible torch/transformers versions")


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.max_length <= 0:
        raise ValueError(f"max_length must be positive, got {args.max_length}")
    if args.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")
    if args.weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {args.weight_decay}")
    if args.projection_hidden_dim is not None and args.projection_hidden_dim <= 0:
        raise ValueError(
            "projection_hidden_dim must be positive when set, "
            f"got {args.projection_hidden_dim}"
        )


def batched_indices(indices: list[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def summarize_tokenization(
    tokenizer,
    examples: list[TextExample],
    indices: list[int],
    *,
    max_length: int,
    batch_size: int = 128,
) -> dict[str, float | int]:
    if not indices:
        raise ValueError("Cannot summarize tokenization for an empty index list.")

    lengths: list[int] = []
    for chunk_indices in batched_indices(indices, batch_size):
        texts = [examples[index].text for index in chunk_indices]
        encoded = tokenizer(
            texts,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    truncation_count = sum(length > max_length for length in lengths)
    return {
        "count": len(lengths),
        "truncation_count": truncation_count,
        "mean_tokens": sum(lengths) / len(lengths),
        "max_tokens": max(lengths),
        "max_length": max_length,
    }


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [item["text"] for item in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if "attention_mask" not in tokenized:
            tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"])
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "targets": torch.stack([item["target"] for item in batch], dim=0),
            "metadata": [item["metadata"] for item in batch],
            "source_indices": [item["source_index"] for item in batch],
        }

    return collate


def mean_norm(tensor: torch.Tensor) -> float:
    validated = validate_target_matrix(tensor, "tensor")
    return validated.norm(p=2, dim=1).mean().item()


def mse_between(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    original_f = validate_target_matrix(original, "original")
    reconstructed_f = validate_target_matrix(reconstructed, "reconstructed")
    if original_f.shape != reconstructed_f.shape:
        raise ValueError(
            f"Shape mismatch: original={original_f.shape} vs "
            f"reconstructed={reconstructed_f.shape}"
        )
    return torch.mean((original_f - reconstructed_f) ** 2).item()


def validation_train_mean_baseline_metrics(
    *,
    train_targets: torch.Tensor,
    validation_targets: torch.Tensor,
) -> dict[str, float]:
    baseline = baseline_mean_reconstruction(train_targets, validation_targets)
    summary = summarize_reconstruction(validation_targets, baseline)
    return {
        "validation_train_mean_baseline_fve": summary["fve"],
        "validation_train_mean_baseline_mse": summary["mse"],
    }


def build_validation_metric_row(
    *,
    epoch: int,
    train_loss: float,
    target_transform_name: str,
    validation_result: ValidationResult,
    train_mean_baseline: dict[str, float],
    is_best: bool,
) -> dict[str, Any]:
    metrics = validation_result.metrics
    return {
        "epoch": epoch,
        "train_mse_loss": train_loss,
        "target_transform": target_transform_name,
        "validation_fve": metrics["fve"],
        "validation_mse": metrics["mse"],
        "validation_rmse": metrics["rmse"],
        "validation_mean_l2_error": metrics["mean_l2_error"],
        "validation_cosine_mean": metrics["cosine_mean"],
        "validation_cosine_std": metrics["cosine_std"],
        "validation_cosine_min": metrics["cosine_min"],
        "validation_cosine_max": metrics["cosine_max"],
        "validation_prediction_norm_mean": mean_norm(
            validation_result.original_predictions
        ),
        "validation_target_norm_mean": mean_norm(
            validation_result.transformed_targets
        ),
        "validation_original_norm_mean": mean_norm(
            validation_result.original_targets
        ),
        "transformed_validation_mse": mse_between(
            validation_result.transformed_targets,
            validation_result.transformed_predictions,
        ),
        **train_mean_baseline,
        "is_best": is_best,
    }


def build_per_example_validation_metrics(
    *,
    metadata_rows: list[dict],
    original_targets: torch.Tensor,
    original_predictions: torch.Tensor,
) -> list[dict]:
    squared_errors = per_example_squared_error(original_targets, original_predictions)
    l2_errors = per_example_l2_error(original_targets, original_predictions)
    cosine = per_example_cosine_similarity(original_targets, original_predictions)
    target_norms = original_targets.detach().float().norm(p=2, dim=1)
    prediction_norms = original_predictions.detach().float().norm(p=2, dim=1)

    rows = []
    for index, metadata in enumerate(metadata_rows):
        rows.append(
            {
                "activation_index": int(metadata.get("activation_index", index)),
                "example_id": metadata.get("example_id"),
                "selected_text_field": metadata.get("ar_text_field"),
                "squared_error": squared_errors[index].item(),
                "l2_error": l2_errors[index].item(),
                "cosine_similarity": cosine[index].item(),
                "target_norm": target_norms[index].item(),
                "prediction_norm": prediction_norms[index].item(),
            }
        )
    return rows


def move_inputs_to_device(
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["targets"].to(device)
    return input_ids, attention_mask, targets


def train_one_epoch(
    *,
    model: TextActivationReconstructor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tqdm,
) -> float:
    model.train()
    if model.freeze_text_model:
        model.text_model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_examples = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        input_ids, attention_mask, targets = move_inputs_to_device(batch, device=device)
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
    model: TextActivationReconstructor,
    dataloader: DataLoader,
    device: torch.device,
    target_transform: TargetTransform,
) -> ValidationResult:
    model.eval()
    prediction_batches = []
    target_batches = []
    metadata_rows = []
    for batch in dataloader:
        input_ids, attention_mask, targets = move_inputs_to_device(batch, device=device)
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction_batches.append(predictions.detach().cpu().float())
        target_batches.append(targets.detach().cpu().float())
        metadata_rows.extend(batch["metadata"])
    if not prediction_batches:
        raise ValueError("Validation dataloader produced no examples.")

    predictions = torch.cat(prediction_batches, dim=0)
    targets = torch.cat(target_batches, dim=0)
    original_predictions = target_transform.inverse_transform(predictions)
    original_targets = target_transform.inverse_transform(targets)
    metrics = summarize_reconstruction(original_targets, original_predictions)
    return ValidationResult(
        metrics=metrics,
        transformed_predictions=predictions,
        transformed_targets=targets,
        original_predictions=original_predictions,
        original_targets=original_targets,
        metadata_rows=metadata_rows,
    )


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in METRIC_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_optional_baseline_metrics(path: str | None) -> Any | None:
    if path is None:
        return None
    baseline_path = Path(path)
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline_metrics_json does not exist: {baseline_path}")
    return json.loads(baseline_path.read_text(encoding="utf-8"))


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


def checkpoint_payload(
    *,
    model: TextActivationReconstructor,
    args: argparse.Namespace,
    epoch: int,
    validation_metrics: dict[str, float],
    activation_artifact: ActivationArtifact,
    target_transform: TargetTransform,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "config": {
            "text_model_name_or_path": args.text_model_name_or_path,
            "activation_dim": activation_artifact.activation_dim,
            "pooling": args.pooling,
            "projection_hidden_dim": args.projection_hidden_dim,
            "dropout": args.dropout,
            "freeze_text_model": args.freeze_text_model,
            "text_field": args.text_field,
            "fallback_text_fields": args.fallback_text_fields,
            "max_length": args.max_length,
            "target_transform": args.target_transform,
        },
        "target_transform_state": target_transform.state_dict_for_checkpoint(),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "activation_artifact_manifest_summary": copy_manifest_summary(
            activation_artifact.manifest
        ),
    }


def save_checkpoint(
    path: Path,
    *,
    model: TextActivationReconstructor,
    args: argparse.Namespace,
    epoch: int,
    validation_metrics: dict[str, float],
    activation_artifact: ActivationArtifact,
    target_transform: TargetTransform,
) -> None:
    torch.save(
        checkpoint_payload(
            model=model,
            args=args,
            epoch=epoch,
            validation_metrics=validation_metrics,
            activation_artifact=activation_artifact,
            target_transform=target_transform,
        ),
        path,
    )


def main() -> None:
    args = parse_args()
    target_transform_name = resolve_target_transform_arg(args)
    validate_args(args)
    set_seed(args.seed)

    activation_dir = Path(args.activation_dir)
    output_dir = Path(args.output_dir)
    fallback_fields = parse_fallback_fields(args.fallback_text_fields)
    training_dtype = dtype_from_arg(args.dtype)
    device = resolve_device(args.device)

    print_section(1, 8, "Loading activation artifact")
    artifact = load_activation_artifact(activation_dir)
    baseline_metrics = load_optional_baseline_metrics(args.baseline_metrics_json)
    print(f"Activations: {tuple(artifact.activations.shape)}")
    print(f"Activation dtype for training targets: {artifact.activations.dtype}")

    print_section(2, 8, "Building AR dataset")
    text_examples = build_text_examples(
        artifact.metadata_rows,
        text_field=args.text_field,
        fallback_text_fields=fallback_fields,
    )
    train_indices, validation_indices, split_strategy = split_train_validation_indices(
        artifact.metadata_rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    prepare_output_dir(output_dir, args.overwrite)
    print(f"Text field: {args.text_field}")
    print(f"Fallback fields: {fallback_fields}")
    print(f"Target transform: {target_transform_name}")
    print(f"Split strategy: {split_strategy}")
    print(f"Train examples: {len(train_indices)}")
    print(f"Validation examples: {len(validation_indices)}")
    text_counts = text_field_counts(text_examples)
    text_lengths = text_length_summary(text_examples)
    print(f"Selected text fields: {text_counts}")
    print(
        "Text length chars: "
        f"mean={text_lengths['mean_chars']:.1f}, max={text_lengths['max_chars']}"
    )
    if args.verbose:
        first = text_examples[0]
        print(f"First example_id: {first.metadata.get('example_id')}")
        print(f"First selected text field: {first.selected_text_field}")

    print_section(3, 8, "Loading text model")
    AutoTokenizer = import_transformers()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name_or_path)
        ensure_tokenizer_padding(tokenizer)
        model = TextActivationReconstructor(
            text_model_name_or_path=args.text_model_name_or_path,
            activation_dim=artifact.activation_dim,
            pooling=args.pooling,
            projection_hidden_dim=args.projection_hidden_dim,
            dropout=args.dropout,
            freeze_text_model=args.freeze_text_model,
        )
    except Exception as exc:
        print_model_loading_error(exc)
        raise
    train_tokenization = summarize_tokenization(
        tokenizer,
        text_examples,
        train_indices,
        max_length=args.max_length,
    )
    validation_tokenization = summarize_tokenization(
        tokenizer,
        text_examples,
        validation_indices,
        max_length=args.max_length,
    )
    model = model.to(device=device)
    if training_dtype != torch.float32:
        model = model.to(dtype=training_dtype)
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(
        "Tokenizer truncation count: "
        f"train={train_tokenization['truncation_count']}, "
        f"validation={validation_tokenization['truncation_count']}"
    )
    print(f"Pooling: {args.pooling}")
    print(f"Freeze text model: {args.freeze_text_model}")
    print(f"Text hidden dim: {model.text_hidden_dim}")
    print(f"Activation dim: {artifact.activation_dim}")

    print_section(4, 8, "Training setup")
    trainable_parameters = count_trainable_parameters(model)
    if trainable_parameters == 0:
        raise ValueError("AR model has no trainable parameters.")
    print(f"Device: {device}")
    print(f"Training dtype: {args.dtype}")
    print(f"Trainable parameters: {trainable_parameters}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    train_original_targets = artifact.activations[train_indices]
    validation_original_targets = artifact.activations[validation_indices]
    target_transform = TargetTransform.fit(target_transform_name, train_original_targets)
    transformed_activations = target_transform.transform(artifact.activations)
    train_mean_baseline = validation_train_mean_baseline_metrics(
        train_targets=train_original_targets,
        validation_targets=validation_original_targets,
    )
    print(
        "Validation train-mean baseline: "
        f"FVE={train_mean_baseline['validation_train_mean_baseline_fve']:.6f}, "
        f"MSE={train_mean_baseline['validation_train_mean_baseline_mse']:.6f}"
    )

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_dataset = ARDataset(
        examples=text_examples,
        activations=transformed_activations,
        indices=train_indices,
    )
    validation_dataset = ARDataset(
        examples=text_examples,
        activations=transformed_activations,
        indices=validation_indices,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    tqdm = import_tqdm()

    print_section(5, 8, "Training")
    metrics_rows: list[dict[str, Any]] = []
    best_fve = float("-inf")
    best_epoch = 0
    best_predictions: torch.Tensor | None = None
    best_targets: torch.Tensor | None = None
    best_metadata_rows: list[dict] | None = None
    best_metrics: dict[str, float] | None = None
    best_per_example_rows: list[dict] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            tqdm=tqdm,
        )
        validation_result = evaluate(
            model=model,
            dataloader=validation_loader,
            device=device,
            target_transform=target_transform,
        )
        validation_metrics = validation_result.metrics
        is_best = validation_metrics["fve"] > best_fve
        if is_best:
            best_fve = validation_metrics["fve"]
            best_epoch = epoch
            best_predictions = validation_result.original_predictions
            best_targets = validation_result.original_targets
            best_metadata_rows = validation_result.metadata_rows
            best_metrics = validation_metrics
            best_per_example_rows = build_per_example_validation_metrics(
                metadata_rows=validation_result.metadata_rows,
                original_targets=validation_result.original_targets,
                original_predictions=validation_result.original_predictions,
            )
            save_checkpoint(
                output_dir / "model.pt",
                model=model,
                args=args,
                epoch=epoch,
                validation_metrics=validation_metrics,
                activation_artifact=artifact,
                target_transform=target_transform,
            )
        if args.save_every_epoch:
            save_checkpoint(
                output_dir / f"model_epoch_{epoch:03d}.pt",
                model=model,
                args=args,
                epoch=epoch,
                validation_metrics=validation_metrics,
                activation_artifact=artifact,
                target_transform=target_transform,
            )
        row = build_validation_metric_row(
            epoch=epoch,
            train_loss=train_loss,
            target_transform_name=target_transform.name,
            validation_result=validation_result,
            train_mean_baseline=train_mean_baseline,
            is_best=is_best,
        )
        metrics_rows.append(row)
        print(
            f"epoch {epoch:03d}: train_mse={train_loss:.6f}, "
            f"val_fve={validation_metrics['fve']:.6f}, "
            f"val_rmse={validation_metrics['rmse']:.6f}, "
            f"val_cosine={validation_metrics['cosine_mean']:.6f}, "
            f"train_mean_fve={train_mean_baseline['validation_train_mean_baseline_fve']:.6f}"
        )

    if best_predictions is None or best_targets is None or best_metadata_rows is None:
        raise ValueError("Training finished without a best validation checkpoint.")
    if best_metrics is None:
        raise ValueError("Training finished without best validation metrics.")
    if best_per_example_rows is None:
        raise ValueError("Training finished without best per-example metrics.")

    print_section(6, 8, "Validation")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation FVE: {best_metrics['fve']:.6f}")
    print(f"Best validation RMSE: {best_metrics['rmse']:.6f}")
    print(f"Best validation cosine mean: {best_metrics['cosine_mean']:.6f}")
    print(
        "Validation train-mean baseline FVE: "
        f"{train_mean_baseline['validation_train_mean_baseline_fve']:.6f}"
    )
    print(
        "Beats validation train-mean baseline: "
        f"{best_metrics['fve'] > train_mean_baseline['validation_train_mean_baseline_fve']}"
    )

    print_section(7, 8, "Writing outputs")
    metrics_csv = output_dir / "training_metrics.csv"
    predictions_path = output_dir / "validation_predictions.pt"
    targets_path = output_dir / "validation_targets.pt"
    metadata_path = output_dir / "validation_metadata.jsonl"
    per_example_path = output_dir / "validation_per_example_metrics.jsonl"
    manifest_path = output_dir / "train_ar_manifest.json"

    write_metrics_csv(metrics_csv, metrics_rows)
    torch.save(best_predictions, predictions_path)
    torch.save(best_targets, targets_path)
    write_jsonl(metadata_path, best_metadata_rows)
    write_jsonl(per_example_path, best_per_example_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "activation_dir": str(activation_dir),
        "output_dir": str(output_dir),
        "activation_artifact_manifest_summary": copy_manifest_summary(artifact.manifest),
        "text_model_name_or_path": args.text_model_name_or_path,
        "text_field": args.text_field,
        "fallback_text_fields": fallback_fields,
        "text_field_counts": text_counts,
        "text_length_summary": text_lengths,
        "tokenization_max_length": args.max_length,
        "tokenization_summary": {
            "train": train_tokenization,
            "validation": validation_tokenization,
        },
        "pooling": args.pooling,
        "projection_hidden_dim": args.projection_hidden_dim,
        "dropout": args.dropout,
        "freeze_text_model": args.freeze_text_model,
        "target_transform": target_transform.name,
        "target_transform_state": target_transform.state_dict_for_manifest(),
        "num_examples": artifact.num_examples,
        "activation_dim": artifact.activation_dim,
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "split_strategy": split_strategy,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "validation_train_mean_baseline": train_mean_baseline,
        "beats_validation_train_mean_baseline": (
            best_metrics["fve"]
            > train_mean_baseline["validation_train_mean_baseline_fve"]
        ),
        "baseline_metrics_json": str(args.baseline_metrics_json)
        if args.baseline_metrics_json
        else None,
        "baseline_metrics": baseline_metrics,
        "output_files": {
            "model": "model.pt",
            "training_metrics": metrics_csv.name,
            "validation_predictions": predictions_path.name,
            "validation_targets": targets_path.name,
            "validation_metadata": metadata_path.name,
            "validation_per_example_metrics": per_example_path.name,
            "manifest": manifest_path.name,
        },
    }
    write_json(manifest_path, manifest)
    for path in (
        output_dir / "model.pt",
        metrics_csv,
        predictions_path,
        targets_path,
        metadata_path,
        per_example_path,
        manifest_path,
    ):
        print(f"Wrote {path}")

    print_section(8, 8, "Result")
    print("SUCCESS: Phase 5b AR diagnostics training completed successfully.")


if __name__ == "__main__":
    main()
