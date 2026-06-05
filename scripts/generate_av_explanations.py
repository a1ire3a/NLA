"""Generate AV explanations from a trained activation verbalizer checkpoint."""

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
from nla_code_interp.models import ActivationVerbalizer  # noqa: E402
from nla_code_interp.utils import set_seed  # noqa: E402
from scripts.train_av import (  # noqa: E402
    build_av_examples,
    ensure_tokenizer_padding,
    generate_rows,
    import_transformers,
    load_activation_artifact,
    parse_fallback_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text explanations from a trained AV checkpoint."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_dir: Path, device: torch.device) -> dict:
    checkpoint_path = checkpoint_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing AV checkpoint file: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive when set, got {args.limit}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {args.max_new_tokens}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    activation_dir = Path(args.activation_dir)
    output_jsonl = Path(args.output_jsonl)
    device = resolve_device()

    checkpoint = load_checkpoint(checkpoint_dir, device)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint {checkpoint_dir / 'model.pt'} is missing config.")

    AutoTokenizer = import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(config["text_model_name_or_path"])
    ensure_tokenizer_padding(tokenizer)
    model = ActivationVerbalizer(
        text_model_name_or_path=config["text_model_name_or_path"],
        activation_dim=int(config["activation_dim"]),
        freeze_lm=bool(config.get("freeze_lm", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    artifact = load_activation_artifact(activation_dir)
    if artifact.activation_dim != int(config["activation_dim"]):
        raise ValueError(
            f"Artifact activation dim {artifact.activation_dim} does not match "
            f"checkpoint activation dim {config['activation_dim']}."
        )

    fallback_fields = parse_fallback_fields(config.get("fallback_text_fields", "code,prompt"))
    examples = build_av_examples(
        artifact.metadata_rows,
        target_text_field=config.get("target_text_field", "reference_description"),
        fallback_text_fields=fallback_fields,
    )
    if args.limit is not None:
        examples = examples[: args.limit]
        activations = artifact.activations[: args.limit]
    else:
        activations = artifact.activations

    rows = generate_rows(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        activations=activations,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(output_jsonl, rows)
    print(f"Wrote {output_jsonl}")
    print("SUCCESS: Phase 7 AV generation completed successfully.")


if __name__ == "__main__":
    main()
