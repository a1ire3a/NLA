# Phase 9a Results: Test Activation Extraction

## Status

Successful.

Activation extraction completed for the three controlled test splits using the main model and the selected context length.

## Setup

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

## Results

| Split | Examples | Shape | Mean | Std | Min | Max | Avg L2 norm | Truncated prompts | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `test_indomain` | 500 | `(500, 1536)` | 0.066030 | 1.287167 | -23.625000 | 18.750000 | 50.499603 | 26 | 13s |
| `test_surface_shift` | 500 | `(500, 1536)` | 0.064251 | 1.275103 | -23.500000 | 18.125000 | 50.018532 | 42 | 14s |
| `test_language_shift` | 361 | `(361, 1536)` | 0.073697 | 1.296118 | -21.750000 | 18.000000 | 50.877804 | 0 | 3s |

## Artifact Outputs

```text
outputs/activations/test_indomain_qwen25_coder_15b_l19_ctx512/
├── activations.pt
├── metadata.jsonl
└── manifest.json

outputs/activations/test_surface_shift_qwen25_coder_15b_l19_ctx512/
├── activations.pt
├── metadata.jsonl
└── manifest.json

outputs/activations/test_language_shift_qwen25_coder_15b_l19_ctx512/
├── activations.pt
├── metadata.jsonl
└── manifest.json
```

These generated artifacts are intentionally not committed to Git.

## Interpretation

The three controlled test artifacts are ready for baseline and full-loop evaluation.

Key points:

1. All extracted tensors have the expected activation dimension, `1536`.
2. In-domain and surface-shift activation statistics are close, which is useful for controlled comparison.
3. Language-shift activations have no truncation in this run.
4. Truncation is low enough for the in-domain and surface-shift splits with `max_length=512`.

## Decision

Proceed to full-loop evaluation on the three test artifacts using the current AV and AR checkpoints.
