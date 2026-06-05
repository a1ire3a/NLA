"""Feasibility probe for CUDA, model loading, and hidden-state extraction."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.activations import (  # noqa: E402
    select_final_non_padding_indices,
    select_token_activations,
    summarize_activation,
)
from nla_code_interp.prompts import code_explanation_prompt_v1  # noqa: E402


SAMPLE_CODE = """\
def count_even(numbers):
    total = 0
    for value in numbers:
        if value % 2 == 0:
            total += 1
    return total
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight feasibility probe for NLA activation extraction."
    )
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument("--layer_index", type=int, default=16, help="Hidden-state tuple index.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum prompt token length.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
        help="torch_dtype value for model loading.",
    )
    parser.add_argument("--device_map", default="auto", help="device_map value for model loading.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True.",
    )
    parser.add_argument("--skip_inputs_embeds_check", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print the probe prompt.")
    return parser.parse_args()


def dtype_from_arg(dtype: str) -> torch.dtype | str:
    mapping: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype]


def print_section(index: int, total: int, label: str) -> None:
    print(f"\n[{index}/{total}] {label}", flush=True)


def gb(num_bytes: int) -> float:
    return num_bytes / 1024**3


def print_vram(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: CUDA unavailable; VRAM stats not available.")
        return

    allocated = gb(torch.cuda.memory_allocated())
    reserved = gb(torch.cuda.memory_reserved())
    max_allocated = gb(torch.cuda.max_memory_allocated())
    print(
        f"{label}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, "
        f"max_allocated={max_allocated:.2f} GB"
    )


def print_environment() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        print(f"GPU index: {device_index}")
        print(f"GPU name: {torch.cuda.get_device_name(device_index)}")
        print(f"GPU capability: {torch.cuda.get_device_capability(device_index)}")
        print_vram("Initial VRAM")
    else:
        print("WARNING: CUDA is unavailable. Continuing, but model loading may fail or run on CPU.")


def import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformers. Install project dependencies with "
            "`pip install -r requirements.txt` inside the project environment."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Loaded model has no parameters.") from exc


def model_layer_count(model: torch.nn.Module) -> int | None:
    config_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if isinstance(config_layers, int):
        return config_layers

    inner_model = getattr(model, "model", None)
    layers = getattr(inner_model, "layers", None)
    if layers is not None:
        return len(layers)
    return None


def print_model_loading_error(kind: str, exc: Exception) -> None:
    print(f"ERROR: {kind} loading failed for the requested model.")
    print(f"Exception: {type(exc).__name__}: {exc}")
    print("Likely causes:")
    print("- model not downloaded or --local_files_only was used before download")
    print("- wrong --model_name_or_path")
    print("- missing Hugging Face access or authentication")
    print("- insufficient VRAM or system RAM")
    print("- incompatible torch/transformers/accelerate versions")


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


def print_activation_summary(summary: dict[str, float]) -> None:
    for key in ("mean", "std", "min", "max", "l2_norm"):
        print(f"activation {key}: {summary[key]:.6f}")


def main() -> None:
    args = parse_args()
    total_sections = 8

    print_section(1, total_sections, "Environment")
    print_environment()

    AutoModelForCausalLM, AutoTokenizer = import_transformers()
    torch_dtype = dtype_from_arg(args.dtype)

    print_section(2, total_sections, "Loading tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        print_model_loading_error("Tokenizer", exc)
        raise
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Tokenizer vocab size: {getattr(tokenizer, 'vocab_size', 'unknown')}")

    print_section(3, total_sections, "Loading model")
    print_vram("Before model load")
    try:
        model = load_causal_lm_with_dtype(
            AutoModelForCausalLM,
            args.model_name_or_path,
            torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        print_model_loading_error("Model", exc)
        raise
    model.eval()
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model dtype argument: {args.dtype}")
    print(f"Model first parameter device: {first_parameter_device(model)}")
    print(f"Model config hidden size: {getattr(model.config, 'hidden_size', 'unknown')}")
    print(f"Model layer count: {model_layer_count(model)}")
    print_vram("After model load")

    print_section(4, total_sections, "Prompt and tokenization")
    prompt = code_explanation_prompt_v1(SAMPLE_CODE)
    if args.verbose:
        print("Prompt:")
        print(prompt)
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        return_attention_mask=True,
    )
    if "attention_mask" not in tokenized:
        tokenized["attention_mask"] = torch.ones_like(tokenized["input_ids"])
    device = first_parameter_device(model)
    tokenized = move_batch_to_device(dict(tokenized), device)
    print(f"input_ids shape: {tuple(tokenized['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(tokenized['attention_mask'].shape)}")
    print(f"non-padding tokens: {int(tokenized['attention_mask'].sum().item())}")

    print_section(5, total_sections, "Hidden-state forward pass")
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    if outputs.hidden_states is None:
        raise ValueError("Model forward pass did not return hidden_states.")
    hidden_states = outputs.hidden_states
    validate_layer_index(args.layer_index, len(hidden_states))
    selected_hidden_state = hidden_states[args.layer_index]
    print(f"number of hidden-state tensors: {len(hidden_states)}")
    print(f"selected layer_index: {args.layer_index}")
    print(f"selected hidden-state shape: {tuple(selected_hidden_state.shape)}")
    print_vram("After hidden-state forward")

    print_section(6, total_sections, "Activation selection")
    token_indices = select_final_non_padding_indices(tokenized["attention_mask"])
    activations = select_token_activations(selected_hidden_state, token_indices)
    summary = summarize_activation(activations)
    print(f"final non-padding token index: {token_indices.detach().cpu().tolist()}")
    print(f"activation shape: {tuple(activations.shape)}")
    print(f"activation dtype: {activations.dtype}")
    print(f"activation device: {activations.device}")
    print_activation_summary(summary)

    print_section(7, total_sections, "inputs_embeds compatibility")
    if args.skip_inputs_embeds_check:
        print("Skipped by --skip_inputs_embeds_check.")
    else:
        embedding_layer = model.get_input_embeddings()
        embedding_device = next(embedding_layer.parameters()).device
        input_ids_for_embedding = tokenized["input_ids"].to(embedding_device)
        with torch.no_grad():
            inputs_embeds = embedding_layer(input_ids_for_embedding)
            embed_outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=tokenized["attention_mask"].to(inputs_embeds.device),
                output_hidden_states=True,
                use_cache=False,
            )
        if embed_outputs.hidden_states is None:
            raise ValueError("inputs_embeds forward pass did not return hidden_states.")
        print("inputs_embeds compatibility check passed.")
        print(f"inputs_embeds shape: {tuple(inputs_embeds.shape)}")
        print_vram("After inputs_embeds forward")

    print_section(8, total_sections, "Result")
    print("SUCCESS: Phase 1 feasibility probe completed successfully.")


if __name__ == "__main__":
    main()
