"""Extract selected hidden-state activations from the target Code LLM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.activations import (  # noqa: E402
    activation_save_dtype,
    select_final_non_padding_indices,
    select_token_activations,
    summarize_activation_batch,
)
from nla_code_interp.data import iter_jsonl, write_jsonl  # noqa: E402
from nla_code_interp.utils import set_seed  # noqa: E402


SCHEMA_VERSION = "phase3_activation_v1"
REQUIRED_INPUT_FIELDS = (
    "example_id",
    "prompt",
    "code",
    "split",
    "language",
    "transformation_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected hidden-state activations from processed NLA JSONL data."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument("--layer_index", type=int, default=16)
    parser.add_argument(
        "--token_position",
        choices=["final_non_padding"],
        default="final_non_padding",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def dtype_from_arg(dtype: str) -> torch.dtype | str:
    mapping: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype]


def import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformers. Install project dependencies with "
            "`pip install -r requirements.txt` inside the project environment."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def import_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        return lambda iterable, **_kwargs: iterable
    return tqdm


def load_causal_lm_with_dtype(
    model_cls,
    model_name_or_path: str,
    dtype: torch.dtype | str,
    **kwargs,
):
    """Load a causal LM with new `dtype=` support and old `torch_dtype=` fallback."""
    try:
        return model_cls.from_pretrained(model_name_or_path, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return model_cls.from_pretrained(model_name_or_path, torch_dtype=dtype, **kwargs)


def print_model_loading_error(kind: str, exc: Exception) -> None:
    print(f"ERROR: {kind} loading failed for the requested model.")
    print(f"Exception: {type(exc).__name__}: {exc}")
    print("Likely causes:")
    print("- model not downloaded or --local_files_only was used before download")
    print("- wrong --model_name_or_path")
    print("- missing Hugging Face access or authentication")
    print("- insufficient VRAM or system RAM")
    print("- incompatible torch/transformers/accelerate versions")


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Loaded model has no parameters.") from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.max_length <= 0:
        raise ValueError(f"max_length must be positive, got {args.max_length}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive when set, got {args.limit}")
    if args.layer_index < 0:
        raise ValueError(f"layer_index must be non-negative, got {args.layer_index}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is non-empty: {output_dir}. "
                "Pass --overwrite to replace it intentionally."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_examples(input_jsonl: Path, limit: int | None) -> list[dict]:
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_jsonl}")
    examples = []
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        if limit is not None and index >= limit:
            break
        validate_example(row, index)
        examples.append(row)
    if not examples:
        raise ValueError(f"No examples loaded from {input_jsonl}")
    return examples


def validate_example(row: dict, index: int) -> None:
    for field in REQUIRED_INPUT_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Example index {index} is missing required non-empty string field {field!r}"
            )


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no pad token and no eos token to reuse for padding.")
    tokenizer.pad_token = tokenizer.eos_token


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def validate_layer_index(layer_index: int, num_hidden_states: int) -> None:
    if layer_index < 0 or layer_index >= num_hidden_states:
        raise ValueError(
            f"Invalid layer_index={layer_index}. Valid hidden-state tuple range is "
            f"0..{num_hidden_states - 1}."
        )


def count_by(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def count_metadata_by(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def batched(examples: list[dict], batch_size: int):
    for start in range(0, len(examples), batch_size):
        yield start, examples[start : start + batch_size]


def token_lengths(tokenizer, prompts: list[str]) -> list[int]:
    encoded = tokenizer(
        prompts,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def tokenize_batch(tokenizer, prompts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if "attention_mask" not in tokenized:
        tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"])
    return dict(tokenized)


def build_metadata_row(
    *,
    row: dict,
    activation_index: int,
    args: argparse.Namespace,
    input_num_tokens: int,
    was_truncated: bool,
    activation_dim: int,
) -> dict:
    metadata_row = dict(row)
    metadata_row.update(
        {
            "activation_index": activation_index,
            "model_name_or_path": args.model_name_or_path,
            "layer_index": args.layer_index,
            "token_position": args.token_position,
            "max_length": args.max_length,
            "input_num_tokens": input_num_tokens,
            "was_truncated": was_truncated,
            "activation_dim": activation_dim,
            "activation_dtype": args.save_dtype,
        }
    )
    return metadata_row


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_artifacts(
    *,
    output_dir: Path,
    activations: torch.Tensor,
    metadata_rows: list[dict],
) -> None:
    if not (output_dir / "activations.pt").exists():
        raise FileNotFoundError("Missing activations.pt after write.")
    if not (output_dir / "metadata.jsonl").exists():
        raise FileNotFoundError("Missing metadata.jsonl after write.")
    if not (output_dir / "manifest.json").exists():
        raise FileNotFoundError("Missing manifest.json after write.")
    if activations.ndim != 2:
        raise ValueError(
            "Expected activations shape [num_examples, hidden_dim], "
            f"got {activations.shape}"
        )
    if activations.shape[0] != len(metadata_rows):
        raise ValueError(
            f"Activation row count {activations.shape[0]} does not match metadata count "
            f"{len(metadata_rows)}."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    save_dtype = activation_save_dtype(args.save_dtype)

    print_section(1, 8, "Loading examples")
    examples = load_examples(input_jsonl, args.limit)
    prepare_output_dir(output_dir, args.overwrite)
    print(f"Loaded examples: {len(examples)}")
    if args.verbose:
        print(f"First example_id: {examples[0]['example_id']}")

    print_section(2, 8, "Loading tokenizer and model")
    AutoModelForCausalLM, AutoTokenizer = import_transformers()
    model_dtype = dtype_from_arg(args.dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        ensure_tokenizer_padding(tokenizer)
        model = load_causal_lm_with_dtype(
            AutoModelForCausalLM,
            args.model_name_or_path,
            model_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        print_model_loading_error("Tokenizer/model", exc)
        raise
    model.eval()
    device = first_parameter_device(model)
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model first parameter device: {device}")
    print(f"Model config hidden size: {getattr(model.config, 'hidden_size', 'unknown')}")

    print_section(3, 8, "Tokenization settings")
    print(f"token_position: {args.token_position}")
    print(f"max_length: {args.max_length}")
    print(f"batch_size: {args.batch_size}")
    print(f"inference dtype: {args.dtype}")
    print(f"save dtype: {args.save_dtype}")

    print_section(4, 8, "Extracting activations")
    tqdm = import_tqdm()
    activation_batches: list[torch.Tensor] = []
    metadata_rows: list[dict] = []
    hidden_dim: int | None = None
    truncation_count = 0

    for _batch_start, batch_rows in tqdm(
        batched(examples, args.batch_size),
        total=(len(examples) + args.batch_size - 1) // args.batch_size,
        desc="batches",
    ):
        prompts = [row["prompt"] for row in batch_rows]
        original_lengths = token_lengths(tokenizer, prompts)
        tokenized = tokenize_batch(tokenizer, prompts, args.max_length)
        tokenized = move_batch_to_device(tokenized, device)

        with torch.no_grad():
            outputs = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        if outputs.hidden_states is None:
            raise ValueError("Model forward pass did not return hidden_states.")
        validate_layer_index(args.layer_index, len(outputs.hidden_states))

        selected_hidden_state = outputs.hidden_states[args.layer_index]
        token_indices = select_final_non_padding_indices(tokenized["attention_mask"])
        batch_activations = select_token_activations(selected_hidden_state, token_indices)
        batch_activations = batch_activations.detach().to(device="cpu", dtype=save_dtype)

        if hidden_dim is None:
            hidden_dim = batch_activations.shape[1]
        elif hidden_dim != batch_activations.shape[1]:
            raise ValueError(
                f"Inconsistent activation dimensions: expected {hidden_dim}, "
                f"got {batch_activations.shape[1]}"
            )

        actual_lengths = tokenized["attention_mask"].sum(dim=1).detach().cpu().tolist()
        for row, original_len, actual_len in zip(
            batch_rows,
            original_lengths,
            actual_lengths,
            strict=True,
        ):
            was_truncated = original_len > actual_len
            truncation_count += int(was_truncated)
            metadata_rows.append(
                build_metadata_row(
                    row=row,
                    activation_index=len(metadata_rows),
                    args=args,
                    input_num_tokens=int(actual_len),
                    was_truncated=was_truncated,
                    activation_dim=batch_activations.shape[1],
                )
            )
        activation_batches.append(batch_activations)

    print_section(5, 8, "Stacking and summarizing")
    activations = torch.cat(activation_batches, dim=0)
    if activations.shape[0] != len(examples):
        raise ValueError(
            f"Extracted {activations.shape[0]} activations for {len(examples)} examples."
        )
    if activations.ndim != 2 or hidden_dim is None or activations.shape[1] != hidden_dim:
        raise ValueError(f"Invalid activation tensor shape: {activations.shape}")
    activation_summary = summarize_activation_batch(activations)
    print(f"Activation tensor shape: {tuple(activations.shape)}")
    for key, value in activation_summary.items():
        print(f"{key}: {value:.6f}")

    print_section(6, 8, "Writing artifacts")
    tensor_file = output_dir / "activations.pt"
    metadata_file = output_dir / "metadata.jsonl"
    manifest_file = output_dir / "manifest.json"
    torch.save(activations, tensor_file)
    write_jsonl(metadata_file, metadata_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cli_args": vars(args),
        "input_path": str(input_jsonl),
        "output_path": str(output_dir),
        "model_name_or_path": args.model_name_or_path,
        "layer_index": args.layer_index,
        "token_position": args.token_position,
        "num_examples": len(examples),
        "hidden_size": getattr(model.config, "hidden_size", None),
        "activation_dim": activations.shape[1],
        "activation_shape": list(activations.shape),
        "activation_dtype": args.save_dtype,
        "tensor_file": tensor_file.name,
        "metadata_file": metadata_file.name,
        "counts_by_split": count_by(examples, "split"),
        "counts_by_language": count_by(examples, "language"),
        "counts_by_transformation_type": count_by(examples, "transformation_type"),
        "truncation_count": truncation_count,
        "activation_summary": activation_summary,
    }
    write_manifest(manifest_file, manifest)
    print(f"Wrote {tensor_file}")
    print(f"Wrote {metadata_file}")
    print(f"Wrote {manifest_file}")

    print_section(7, 8, "Verification")
    verify_artifacts(output_dir=output_dir, activations=activations, metadata_rows=metadata_rows)
    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Truncated prompts: {truncation_count}")
    print(f"Counts by split: {count_metadata_by(metadata_rows, 'split')}")

    print_section(8, 8, "Result")
    print("SUCCESS: Phase 3 activation extraction completed successfully.")


if __name__ == "__main__":
    main()
