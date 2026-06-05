"""Train a Qwen-based aligned Activation Verbalizer with LoRA."""

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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.data import write_jsonl  # noqa: E402
from nla_code_interp.qwen_models import (  # noqa: E402
    DEFAULT_QWEN_MODEL,
    LoraSettings,
    QwenActivationVerbalizer,
    apply_lora,
    dtype_from_name,
    load_qwen_causal_lm,
    qwen_checkpoint_metadata,
    qwen_trainable_parameter_summary,
)
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.train_av import (  # noqa: E402
    ActivationArtifact,
    build_av_examples,
    ensure_tokenizer_padding,
    import_tqdm,
    import_transformers,
    load_activation_artifact,
    target_field_counts,
    text_length_summary,
)
from scripts.train_ar import resolve_device  # noqa: E402


SCHEMA_VERSION = "phase10a_qwen_av_lora_v1"
TARGET_TEXT_FIELDS = ("reference_description", "code", "prompt")
METRIC_COLUMNS = ("epoch", "train_loss", "validation_loss", "is_best")


@dataclass(frozen=True)
class ActivationTextExample:
    activation: torch.Tensor
    target_text: str
    metadata: dict[str, Any]


class QwenAVDataset(Dataset):
    def __init__(self, examples: list[ActivationTextExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ActivationTextExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA activation-to-text AV.")
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--validation_activation_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--target_text_field",
        choices=TARGET_TEXT_FIELDS,
        default="reference_description",
    )
    parser.add_argument("--fallback_text_fields", default="prompt,code")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_target_length", type=int, default=128)
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
    invalid = [field for field in fields if field not in TARGET_TEXT_FIELDS]
    if invalid:
        raise ValueError(f"Invalid fallback target field(s): {invalid}")
    return fields


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
    target_text_field: str,
    fallback_text_fields: list[str],
) -> list[ActivationTextExample]:
    av_examples = build_av_examples(
        artifact.metadata_rows,
        target_text_field=target_text_field,
        fallback_text_fields=fallback_text_fields,
    )
    examples = []
    for index, av_example in enumerate(av_examples):
        metadata = dict(av_example.metadata)
        metadata["qwen_av_target_field"] = av_example.selected_target_field
        metadata["qwen_av_target_text"] = av_example.target_text
        examples.append(
            ActivationTextExample(
                activation=artifact.activations[index],
                target_text=av_example.target_text,
                metadata=metadata,
            )
        )
    return examples


def append_eos(text: str, eos_token: str | None) -> str:
    if not eos_token or text.endswith(eos_token):
        return text
    return text + eos_token


def make_collate_fn(tokenizer, max_target_length: int):
    def collate(batch: list[ActivationTextExample]) -> dict[str, Any]:
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
            "input_ids": tokenized["input_ids"],
            "attention_mask": attention_mask,
            "metadata": [item.metadata for item in batch],
            "target_texts": [item.target_text for item in batch],
        }

    return collate


def move_batch(
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
    model: QwenActivationVerbalizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tqdm,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        activations, input_ids, attention_mask = move_batch(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            activations=activations,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        if outputs.loss is None:
            raise ValueError("Qwen AV model did not return a training loss.")
        outputs.loss.backward()
        optimizer.step()
        batch_size = activations.shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("Training dataloader produced no examples.")
    return total_loss / total_examples


@torch.no_grad()
def evaluate_loss(
    *,
    model: QwenActivationVerbalizer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        activations, input_ids, attention_mask = move_batch(batch, device=device)
        outputs = model(
            activations=activations,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        if outputs.loss is None:
            raise ValueError("Qwen AV model did not return a validation loss.")
        batch_size = activations.shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("Validation dataloader produced no examples.")
    return total_loss / total_examples


@torch.no_grad()
def generate_rows(
    *,
    model: QwenActivationVerbalizer,
    tokenizer,
    examples: list[ActivationTextExample],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    tqdm=None,
    desc: str = "generate",
    log_every_batches: int = 0,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    starts = range(0, len(examples), batch_size)
    iterator = tqdm(starts, desc=desc, leave=False) if tqdm is not None else starts
    total_batches = (len(examples) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(iterator, start=1):
        batch_examples = examples[start : start + batch_size]
        activations = torch.stack([item.activation for item in batch_examples], dim=0).to(device)
        generated_ids = model.greedy_generate(
            activations=activations,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_texts = tokenizer.batch_decode(
            generated_ids.detach().cpu(),
            skip_special_tokens=True,
        )
        for offset, (example, generated_text) in enumerate(
            zip(batch_examples, generated_texts, strict=True)
        ):
            metadata = example.metadata
            rows.append(
                {
                    "activation_index": int(metadata.get("activation_index", start + offset)),
                    "example_id": metadata.get("example_id"),
                    "target_text": example.target_text,
                    "generated_text": generated_text.strip(),
                    "split": metadata.get("split"),
                    "language": metadata.get("language"),
                    "transformation_type": metadata.get("transformation_type"),
                }
            )
        if log_every_batches and (
            batch_number % log_every_batches == 0 or batch_number == total_batches
        ):
            print(
                f"{desc}: {batch_number}/{total_batches} batches "
                f"({len(rows)}/{len(examples)} examples)",
                flush=True,
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


def save_qwen_parts(
    *,
    model: QwenActivationVerbalizer,
    tokenizer,
    output_dir: Path,
    lora_settings: LoraSettings,
) -> dict[str, str]:
    files = {"activation_projection": "activation_projection.pt", "tokenizer": "tokenizer"}
    tokenizer.save_pretrained(output_dir / "tokenizer")
    torch.save(
        model.activation_projection.state_dict(),
        output_dir / files["activation_projection"],
    )
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
    epoch: int,
    validation_loss: float,
    output_files: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": qwen_checkpoint_metadata(
            component="qwen_av",
            model_name_or_path=args.model_name_or_path,
            activation_dim=activation_dim,
            dtype=args.dtype,
            lora_settings=lora_settings,
            extra_config={
                "target_text_field": args.target_text_field,
                "fallback_text_fields": args.fallback_text_fields,
                "max_target_length": args.max_target_length,
            },
        ),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "output_files": output_files,
    }


def save_checkpoint(
    *,
    output_dir: Path,
    model: QwenActivationVerbalizer,
    tokenizer,
    args: argparse.Namespace,
    activation_dim: int,
    lora_settings: LoraSettings,
    epoch: int,
    validation_loss: float,
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
        epoch=epoch,
        validation_loss=validation_loss,
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

    print_section(2, 8, "Preparing target text")
    train_av_examples_for_stats = build_av_examples(
        train_artifact.metadata_rows,
        target_text_field=args.target_text_field,
        fallback_text_fields=fallback_fields,
    )
    train_examples = build_examples(
        artifact=train_artifact,
        target_text_field=args.target_text_field,
        fallback_text_fields=fallback_fields,
    )
    validation_examples = build_examples(
        artifact=validation_artifact,
        target_text_field=args.target_text_field,
        fallback_text_fields=fallback_fields,
    )
    print(f"Train target fields: {target_field_counts(train_av_examples_for_stats)}")
    print(f"Train target lengths: {text_length_summary(train_av_examples_for_stats)}")

    print_section(3, 8, "Loading tokenizer and Qwen AV")
    AutoTokenizer = import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    ensure_tokenizer_padding(tokenizer)
    qwen_model = load_qwen_causal_lm(
        model_name_or_path=args.model_name_or_path,
        dtype=dtype_from_name(args.dtype),
    )
    qwen_model = apply_lora(qwen_model, lora_settings=lora_settings)
    model = QwenActivationVerbalizer(
        qwen_model=qwen_model,
        activation_dim=train_artifact.activation_dim,
    ).to(device)
    print(f"Model: {args.model_name_or_path}")
    print(f"LoRA: {lora_settings.as_dict()}")
    print(f"Parameters: {qwen_trainable_parameter_summary(model)}")
    print(f"Device: {device}")

    print_section(4, 8, "Training setup")
    collate_fn = make_collate_fn(tokenizer, args.max_target_length)
    train_loader = DataLoader(
        QwenAVDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        QwenAVDataset(validation_examples),
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
    best_loss = float("inf")
    best_epoch = 0
    output_files: dict[str, str] = {}
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
        is_best = validation_loss < best_loss
        if is_best:
            best_loss = validation_loss
            best_epoch = epoch
            output_files = save_checkpoint(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                args=args,
                activation_dim=train_artifact.activation_dim,
                lora_settings=lora_settings,
                epoch=epoch,
                validation_loss=validation_loss,
            )
        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "is_best": is_best,
            }
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f}, "
            f"validation_loss={validation_loss:.6f}"
        )

    print_section(6, 8, "Generating validation examples")
    generation_rows = generate_rows(
        model=model,
        tokenizer=tokenizer,
        examples=validation_examples,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_target_length,
    )
    print(f"Generated validation rows: {len(generation_rows)}")

    print_section(7, 8, "Writing outputs")
    metrics_path = output_dir / "training_metrics.csv"
    generations_path = output_dir / "validation_generations.jsonl"
    manifest_path = output_dir / "train_qwen_av_manifest.json"
    write_metrics_csv(metrics_path, metrics_rows)
    write_jsonl(generations_path, generation_rows)
    output_files.update(
        {
            "training_metrics": metrics_path.name,
            "validation_generations": generations_path.name,
            "manifest": manifest_path.name,
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "activation_dir": args.activation_dir,
        "validation_activation_dir": args.validation_activation_dir,
        "output_dir": args.output_dir,
        "model_name_or_path": args.model_name_or_path,
        "target_text_field": args.target_text_field,
        "fallback_text_fields": fallback_fields,
        "lora": lora_settings.as_dict(),
        "dtype": args.dtype,
        "train_count": train_artifact.num_examples,
        "validation_count": validation_artifact.num_examples,
        "activation_dim": train_artifact.activation_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "output_files": output_files,
    }
    write_json(manifest_path, manifest)
    for filename in output_files.values():
        print(f"Wrote {output_dir / filename}")

    print_section(8, 8, "Result")
    print("SUCCESS: Phase 10a Qwen AV training completed successfully.")


if __name__ == "__main__":
    main()
