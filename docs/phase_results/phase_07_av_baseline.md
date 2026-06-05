# Phase 7 Results: AV Baseline

## Status

Successful.

The first supervised Activation Verbalizer (AV) was trained from saved activation vectors to reference-description text targets. Validation generation also completed successfully.

## Setup

| Field | Value |
|---|---|
| Train artifact | `outputs/activations/train_qwen25_coder_15b_l19_ctx512` |
| Validation artifact | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| Text model | `distilgpt2` |
| Target text field | `reference_description` |
| Fallback fields | `code,prompt` |
| Train examples | 5000 |
| Validation examples | 500 |
| Activation dim | 1536 |
| LM embedding dim | 768 |
| LM frozen | false |
| Epochs | 5 |
| Batch size | 8 |
| Learning rate | 5e-5 |
| Max target length | 64 |

## Target Text Statistics

| Split | Field count | Mean chars | Min chars | Max chars |
|---|---|---:|---:|---:|
| train | `reference_description`: 5000 | 254.1486 | 4 | 3111 |
| validation | `reference_description`: 500 | 294.7720 | 10 | 2831 |

## Training Result

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 3.292862 | 3.161037 |
| 2 | 2.926588 | 3.106717 |
| 3 | 2.734130 | 3.098751 |
| 4 | 2.580585 | 3.129976 |
| 5 | 2.444565 | 3.165976 |

Best validation loss occurred at epoch 3.

## Generation Result

Validation generation completed successfully.

Generated rows:

```text
500
```

A separate generation command also completed successfully for 50 validation examples:

```text
outputs/reports/av/validation_refdesc_distilgpt2_generations.jsonl
```

## Output Files

Generated outputs were written locally under ignored `outputs/` paths:

```text
outputs/checkpoints/av/train5000_val500_qwen25_coder_15b_l19_ctx512_refdesc_distilgpt2/model.pt
outputs/checkpoints/av/train5000_val500_qwen25_coder_15b_l19_ctx512_refdesc_distilgpt2/training_metrics.csv
outputs/checkpoints/av/train5000_val500_qwen25_coder_15b_l19_ctx512_refdesc_distilgpt2/validation_generations.jsonl
outputs/checkpoints/av/train5000_val500_qwen25_coder_15b_l19_ctx512_refdesc_distilgpt2/train_av_manifest.json
outputs/reports/av/validation_refdesc_distilgpt2_generations.jsonl
```

## Interpretation

This AV baseline is sufficient for moving to the first full NLA loop.

Important observations:

1. The activation-to-text training pipeline works end-to-end.
2. Validation loss improves through epoch 3 and then worsens slightly, suggesting mild overfitting after that point.
3. The model generated validation outputs successfully, so AV output artifacts are available for downstream AR-based reconstruction.
4. This is not the final AV architecture; it is a supervised baseline to complete the first vector-to-text-to-vector pipeline.

## Decision

Proceed to Phase 8: connect the full NLA loop.

The next step should evaluate:

```text
activation -> AV generated explanation -> AR -> reconstructed activation -> FVE
```

using the current best AR checkpoint and the AV-generated validation explanations.
