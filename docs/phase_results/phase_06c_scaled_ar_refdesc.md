# Phase 6c Results: Scaled AR Training — Reference Description

## Status

Successful.

The AR model was trained on the scaled train artifact and evaluated on the external validation artifact. This is the first larger-data AR run after the pilot diagnostics.

## Setup

| Field | Value |
|---|---|
| Train artifact | `outputs/activations/train_qwen25_coder_15b_l19_ctx512` |
| Validation artifact | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| Text model | `distilbert-base-uncased` |
| Text field | `reference_description` |
| Fallback fields | `prompt,code` |
| Target transform | `standardize` |
| Text encoder | frozen |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.001 |
| AR text max length | 256 |

## Baseline Reference

Validation train-mean baseline:

| Metric | Value |
|---|---:|
| FVE | -0.005988 |
| MSE | 0.235964 |

## Result

| Metric | Value |
|---|---:|
| Best epoch | 12 |
| Best validation FVE | 0.095719 |
| Best validation RMSE | 0.460551 |
| Best validation cosine mean | 0.925471 |
| Beats train-mean baseline | yes |

## Notes

Tokenizer truncation was reported for the AR text encoder:

| Split | Truncated examples |
|---|---:|
| train | 166 |
| validation | 24 |

This truncation is for the DistilBERT AR text input, not for the original Qwen activation extraction. The Qwen activation extraction context remains `ctx512`.

## Interpretation

The larger-data AR result is positive and useful:

1. Validation FVE is positive: `0.095719`.
2. The model clearly beats the validation train-mean baseline of `-0.005988`.
3. The best epoch is 12, after which validation FVE fluctuates rather than consistently improving.
4. The result confirms that standardized targets remain the right default for AR.
5. The next comparison should test whether using source code text improves reconstruction over `reference_description`.

## Decision

Proceed with a code-text AR comparison on the same train/validation artifacts.

Recommended next run:

```text
text_field = code
target_transform = standardize
max_length = 512
freeze_text_model = true
```
