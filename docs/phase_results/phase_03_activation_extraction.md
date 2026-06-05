# Phase 3 Results: Activation Extraction

## Status

**Accepted / successful.**

The activation extraction pipeline successfully produced activation artifacts for the pilot dataset using both the smoke-test model and the main model. A follow-up context-length ablation on the main model showed that longer context lengths sharply reduce prompt truncation without causing runtime issues on the available GPU.

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

### Pilot extraction: 100 examples, 1.5B model, context 128

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

### Context-length ablation: 100 examples, 1.5B model, context 256

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/pilot_100.jsonl \
  --output_dir outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx256 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 256 \
  --batch_size 8 \
  --dtype bfloat16 \
  --seed 42
```

### Context-length ablation: 100 examples, 1.5B model, context 512

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/pilot_100.jsonl \
  --output_dir outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx512 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 512 \
  --batch_size 8 \
  --dtype bfloat16 \
  --seed 42
```

### Context-length ablation: 100 examples, 1.5B model, context 1024

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/pilot_100.jsonl \
  --output_dir outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx1024 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 1024 \
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

outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx256/
├── activations.pt
├── metadata.jsonl
└── manifest.json

outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx512/
├── activations.pt
├── metadata.jsonl
└── manifest.json

outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx1024/
├── activations.pt
├── metadata.jsonl
└── manifest.json
```

These generated artifacts are intentionally not committed to Git.

## Results

| Run | Model | Layer | Max length | Examples | Activation shape | Mean | Std | Min | Max | Avg L2 norm | Truncated prompts | Status |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| Smoke | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 16 | 128 | 10 | `(10, 896)` | 0.013654 | 0.840399 | -11.937500 | 13.687500 | 25.134390 | 8 | Success |
| Pilot | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 16 | 128 | 100 | `(100, 896)` | 0.021549 | 0.868290 | -12.437500 | 14.437500 | 25.973282 | 66 | Success |
| Pilot | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 19 | 128 | 100 | `(100, 1536)` | 0.029074 | 1.275952 | -24.125000 | 18.375000 | 49.950233 | 66 | Success |
| Context ablation | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 19 | 256 | 100 | `(100, 1536)` | 0.050653 | 1.275938 | -23.625000 | 18.375000 | 50.001606 | 30 | Success |
| Context ablation | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 19 | 512 | 100 | `(100, 1536)` | 0.067766 | 1.287972 | -20.625000 | 18.250000 | 50.543163 | 1 | Success |
| Context ablation | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 19 | 1024 | 100 | `(100, 1536)` | 0.068601 | 1.288932 | -20.625000 | 18.250000 | 50.584785 | 0 | Success |

## Interpretation

Phase 3 validates that the project can move from individual hidden-state probing to dataset-level activation extraction.

Key conclusions:

1. The extraction pipeline works on processed JSONL data.
2. The 0.5B model produces pilot activations with shape `[100, 896]`.
3. The 1.5B model produces pilot activations with shape `[100, 1536]`.
4. Metadata and tensor row counts match.
5. Artifact format is ready for baseline reconstruction and AR training.
6. Context length has a large effect on truncation for function-level code prompts.

## Context-Length Ablation Decision

With `max_length=128`, 66 out of 100 pilot prompts were truncated.

The main-model ablation shows:

- `max_length=256`: 30 truncated prompts.
- `max_length=512`: 1 truncated prompt.
- `max_length=1024`: 0 truncated prompts.

Because `max_length=512` nearly eliminates truncation and is almost as stable as `max_length=1024`, it is a good default for main-model pilot and larger extractions. `max_length=1024` is the cleanest no-truncation option and can be used for final pilot evidence or if runtime remains acceptable on larger splits.

For the next baseline phase, the recommended primary artifact is:

```text
outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx512
```

The `ctx1024` artifact should be kept as a no-truncation comparison point.

## Decision

Proceed to **Phase 4: Metrics and Baselines** before extracting the full dataset.

Rationale: Phase 4 validates FVE, baseline behavior, artifact loading, and reporting on small pilot artifacts. Full extraction over train/validation/test should happen after this validation step, using the selected context length.
