# Phase 3 Results: Activation Extraction

## Status

**Accepted / successful.**

The activation extraction pipeline successfully produced activation artifacts for the pilot dataset using both the smoke-test model and the main model.

## Code Review Summary

The implemented pipeline in `scripts/extract_activations.py`:

- Reads processed JSONL examples.
- Validates required fields.
- Loads a Hugging Face tokenizer and causal LM.
- Uses the same `dtype=` / `torch_dtype=` compatibility strategy as the feasibility probe.
- Tokenizes prompts in batches with truncation and padding.
- Runs a no-grad forward pass with `output_hidden_states=True`.
- Selects the final non-padding token activation from the selected hidden-state layer.
- Saves all activations as one CPU tensor in `activations.pt`.
- Saves one metadata row per activation in `metadata.jsonl`.
- Saves run metadata and summary statistics in `manifest.json`.
- Verifies that tensor and metadata row counts match.

The helper module `src/nla_code_interp/activations.py` now includes:

- `activation_save_dtype`
- `summarize_activation_batch`
- existing final-token selection utilities

Lightweight tests were added in `tests/test_activation_io.py` for dtype mapping, batch summaries, and synthetic artifact row-count checks.

## Commands Run

### Smoke extraction: 10 examples, 0.5B model

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/pilot_100.jsonl \
  --output_dir outputs/activations/pilot_100_qwen25_coder_05b_l16_smoke10 \
  --model_name_or_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --layer_index 16 \
  --max_length 128 \
  --batch_size 8 \
  --dtype bfloat16 \
  --limit 10 \
  --seed 42
```

### Pilot extraction: 100 examples, 0.5B model

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/pilot_100.jsonl \
  --output_dir outputs/activations/pilot_100_qwen25_coder_05b_l16 \
  --model_name_or_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --layer_index 16 \
  --max_length 128 \
  --batch_size 8 \
  --dtype bfloat16 \
  --seed 42
```

### Pilot extraction: 100 examples, 1.5B model

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/pilot_100.jsonl \
  --output_dir outputs/activations/pilot_100_qwen25_coder_15b_l19 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 128 \
  --batch_size 8 \
  --dtype bfloat16 \
  --seed 42
```

## Artifact Outputs

```text
outputs/activations/pilot_100_qwen25_coder_05b_l16_smoke10/
├── activations.pt
├── metadata.jsonl
└── manifest.json

outputs/activations/pilot_100_qwen25_coder_05b_l16/
├── activations.pt
├── metadata.jsonl
└── manifest.json

outputs/activations/pilot_100_qwen25_coder_15b_l19/
├── activations.pt
├── metadata.jsonl
└── manifest.json
```

These generated artifacts are intentionally not committed to Git.

## Results

| Run | Model | Layer | Examples | Activation shape | Mean | Std | Min | Max | Avg L2 norm | Truncated prompts | Status |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| Smoke | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 16 | 10 | `(10, 896)` | 0.013654 | 0.840399 | -11.937500 | 13.687500 | 25.134390 | 8 | Success |
| Pilot | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 16 | 100 | `(100, 896)` | 0.021549 | 0.868290 | -12.437500 | 14.437500 | 25.973282 | 66 | Success |
| Pilot | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 19 | 100 | `(100, 1536)` | 0.029074 | 1.275952 | -24.125000 | 18.375000 | 49.950233 | 66 | Success |

## Interpretation

Phase 3 validates that the project can move from individual hidden-state probing to dataset-level activation extraction.

Key conclusions:

1. The extraction pipeline works on processed JSONL data.
2. The 0.5B model produces pilot activations with shape `[100, 896]`.
3. The 1.5B model produces pilot activations with shape `[100, 1536]`.
4. Metadata and tensor row counts match.
5. Artifact format is ready for baseline reconstruction and AR training.

## Important Note: Prompt Truncation

With `max_length=128`, 66 out of 100 pilot prompts were truncated for both full pilot runs.

This is not a blocker for Phase 3, but it is scientifically important. Future phases should either:

- keep `max_length=128` and document it as an intentional compute-saving constraint, or
- run a comparison extraction at `max_length=256` on the pilot set before scaling to larger splits.

For Phase 4 baselines, the current pilot artifacts are sufficient.

## Decision

Proceed to **Phase 4: Metrics and Baselines**.

The next phase should implement FVE evaluation and simple reconstruction baselines over saved activation artifacts before training AR or AV.
