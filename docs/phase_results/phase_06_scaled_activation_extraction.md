# Phase 6 Results: Scaled Activation Extraction

## Status

**Successful.**

Main-model train and validation activations were extracted with the selected context length, `max_length=512`. These artifacts are now ready for baseline evaluation and larger AR training.

## Model and Extraction Setup

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Layer index | `19` |
| Token position | `final_non_padding` |
| Max length | `512` |
| Batch size | `8` |
| Inference dtype | `bfloat16` |
| Saved activation dtype | `float32` |
| Activation dimension | `1536` |

## Commands Run

### Train split

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/train.jsonl \
  --output_dir outputs/activations/train_qwen25_coder_15b_l19_ctx512 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 512 \
  --batch_size 8 \
  --dtype bfloat16 \
  --seed 42
```

### Validation split

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/validation.jsonl \
  --output_dir outputs/activations/validation_qwen25_coder_15b_l19_ctx512 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 512 \
  --batch_size 8 \
  --dtype bfloat16 \
  --seed 42
```

## Results

| Split | Examples | Batches | Runtime | Activation shape | Mean | Std | Min | Max | Avg L2 norm | Truncated prompts | Truncation rate |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5000 | 625 | 2m24s | `(5000, 1536)` | 0.065013 | 1.285532 | -26.500000 | 21.125000 | 50.429317 | 368 | 7.36% |
| validation | 500 | 63 | 15s | `(500, 1536)` | 0.063659 | 1.283578 | -24.875000 | 20.500000 | 50.340824 | 51 | 10.20% |

## Artifact Outputs

```text
outputs/activations/train_qwen25_coder_15b_l19_ctx512/
├── activations.pt
├── metadata.jsonl
└── manifest.json

outputs/activations/validation_qwen25_coder_15b_l19_ctx512/
├── activations.pt
├── metadata.jsonl
└── manifest.json
```

These generated artifacts are intentionally not committed to Git.

## Interpretation

The extraction results are consistent with the pilot context-length decision:

1. `max_length=512` works for larger train/validation extraction.
2. The activation dimensions are correct and stable at `1536`.
3. Train and validation activation statistics are very similar.
4. Truncation is much lower than the original pilot `max_length=128` setting, but not zero.
5. The resulting artifacts are large enough to train a more meaningful AR model.

## Decision

Proceed with:

1. Baseline evaluation for train and validation artifacts.
2. AR training on the larger train/validation artifacts using `target_transform=standardize`.
3. A comparison run using `text_field=code` after the reference-description AR run.
