"""Generate explanations from a trained Qwen Activation Verbalizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nla_code_interp.data import write_jsonl  # noqa: E402
from nla_code_interp.qwen_models import (  # noqa: E402
    QwenActivationVerbalizer,
    dtype_from_name,
    load_qwen_causal_lm,
)
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.train_av import ensure_tokenizer_padding, import_transformers  # noqa: E402
from scripts.train_qwen_av import (  # noqa: E402
    build_examples,
    generate_rows,
    parse_fallback_fields,
    subset_artifact,
)
from scripts.train_av import load_activation_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text explanations from a Qwen AV checkpoint."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive when set, got {args.limit}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {args.max_new_tokens}")


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_dir: Path, device: torch.device) -> dict:
    checkpoint_path = checkpoint_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Qwen AV checkpoint file: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict in {checkpoint_path}, got {type(checkpoint)}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{checkpoint_path} is missing config.")
    output_files = checkpoint.get("output_files")
    if not isinstance(output_files, dict):
        raise ValueError(f"{checkpoint_path} is missing output_files.")
    return checkpoint


def load_qwen_av_model(
    *,
    checkpoint_dir: Path,
    checkpoint: dict,
    device: torch.device,
) -> tuple[QwenActivationVerbalizer, object, dict]:
    config = checkpoint["config"]
    output_files = checkpoint["output_files"]
    model_name_or_path = config["model_name_or_path"]
    dtype_name = config.get("dtype", "bfloat16")

    AutoTokenizer = import_transformers()
    tokenizer_path = checkpoint_dir / output_files.get("tokenizer", "tokenizer")
    tokenizer_source = tokenizer_path if tokenizer_path.exists() else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    ensure_tokenizer_padding(tokenizer)

    qwen_model = load_qwen_causal_lm(
        model_name_or_path=model_name_or_path,
        dtype=dtype_from_name(dtype_name),
    )
    lora_config = config.get("lora", {})
    if lora_config.get("enabled", False):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Could not import peft. Install project dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        adapter_dir = checkpoint_dir / output_files.get("qwen_adapter", "qwen_adapter")
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Missing Qwen AV adapter directory: {adapter_dir}")
        qwen_model = PeftModel.from_pretrained(qwen_model, adapter_dir)
    elif "qwen_model_state" in output_files:
        state_path = checkpoint_dir / output_files["qwen_model_state"]
        qwen_model.load_state_dict(torch.load(state_path, map_location="cpu"))

    model = QwenActivationVerbalizer(
        qwen_model=qwen_model,
        activation_dim=int(config["activation_dim"]),
    )
    projection_path = checkpoint_dir / output_files["activation_projection"]
    model.activation_projection.load_state_dict(
        torch.load(projection_path, map_location="cpu")
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer, config


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    activation_dir = Path(args.activation_dir)
    output_jsonl = Path(args.output_jsonl)
    device = resolve_device()

    checkpoint = load_checkpoint(checkpoint_dir, device)
    model, tokenizer, config = load_qwen_av_model(
        checkpoint_dir=checkpoint_dir,
        checkpoint=checkpoint,
        device=device,
    )
    artifact = subset_artifact(load_activation_artifact(activation_dir), args.limit)
    if artifact.activation_dim != int(config["activation_dim"]):
        raise ValueError(
            f"Artifact activation dim {artifact.activation_dim} does not match "
            f"checkpoint activation dim {config['activation_dim']}."
        )
    fallback_fields = parse_fallback_fields(config.get("fallback_text_fields", "prompt,code"))
    examples = build_examples(
        artifact=artifact,
        target_text_field=config.get("target_text_field", "reference_description"),
        fallback_text_fields=fallback_fields,
    )
    rows = generate_rows(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(output_jsonl, rows)
    print(f"Wrote {output_jsonl}")
    print("SUCCESS: Phase 10a Qwen AV generation completed successfully.")


if __name__ == "__main__":
    main()
