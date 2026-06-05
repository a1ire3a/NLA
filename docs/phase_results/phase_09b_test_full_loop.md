# Phase 9b Results: Test Full-Loop Evaluation

## Status

Successful execution; reconstruction quality remains below the mean baseline.

The current full NLA loop was evaluated on the three controlled test activation artifacts.

## Setup

| Field | Value |
|---|---|
| AV checkpoint | `outputs/checkpoints/av/train5000_val500_qwen25_coder_15b_l19_ctx512_refdesc_distilgpt2` |
| AR checkpoint | `outputs/checkpoints/ar/train5000_val500_qwen25_coder_15b_l19_ctx512_code_distilbert_standardize` |
| AV training target | `reference_description` |
| AR training text field | `code` |
| Max generated tokens | 64 |
| Batch size | 8 |

## Results

| Split | Examples | NLA FVE | NLA MSE | Mean FVE | Mean MSE | Zero FVE | Shuffled FVE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_indomain` | 500 | -0.758254 | 0.232325 | 0.000000 | 0.132134 | -11.571788 | -0.929892 |
| `test_surface_shift` | 500 | -0.505704 | 0.297379 | 0.000000 | 0.197502 | -7.253178 | -0.997270 |
| `test_language_shift` | 361 | -13.048594 | 0.268444 | 0.000000 | 0.019108 | -87.200211 | -0.945287 |

## Interpretation

The test full-loop evaluation confirms the same pattern observed on validation:

1. The full-loop implementation works on all controlled test splits.
2. The current NLA loop is worse than the mean baseline on every split.
3. The current NLA loop is better than the zero baseline on every split.
4. It is better than the shuffled baseline on the surface-shift and language-shift splits, but slightly worse than shuffled on the in-domain split.
5. The language-shift FVE is extremely negative because the mean baseline MSE is very small (`0.019108`), so the denominator of FVE is small. The raw NLA MSE is similar to the other splits, but relative to the language-shift variance it is poor.

## Main Bottleneck

The current bottleneck is not pipeline execution. The pipeline is complete.

The bottleneck is that the supervised AV-generated explanations do not preserve enough activation-specific information for AR reconstruction.

This is expected for the first supervised baseline because:

- AV was trained to imitate `reference_description`, not to maximize reconstruction.
- AR was strongest when trained on `code`, but the full loop gives it generated natural-language text.
- No reconstruction-driven or joint AV-AR objective has been used yet.

## Decision

Proceed to reconstruction-driven improvement rather than repeating the same supervised baseline.

The next implementation should add a Phase 9c / Phase 10-style improvement path:

```text
activation -> AV generated text -> AR -> reconstructed activation -> reconstruction loss
```

At minimum, implement evaluation support for using AV-generated explanations as AR training data, or implement an alternating / reconstruction-driven fine-tuning stage.
