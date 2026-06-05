# Phase 6c Results: Scaled AR Training — Code Text

## Status

Successful.

The AR model was trained on the scaled train artifact and evaluated on the external validation artifact using source code as the input text. This run outperformed the reference-description AR run.

## Setup

| Field | Value |
|---|---|
| Train artifact | `outputs/activations/train_qwen25_coder_15b_l19_ctx512` |
| Validation artifact | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| Text model | `distilbert-base-uncased` |
| Text field | `code` |
| Target transform | `standardize` |
| Text encoder | frozen |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.001 |
| AR text max length | 512 |

## Text Statistics

| Split | Selected text field | Mean chars | Max chars | Tokenizer truncation count |
|---|---|---:|---:|---:|
| train | `code` | 880.2 | 3946 | 520 |
| validation | `code` | 951.9 | 3925 | 60 |

## Baseline Reference

Validation train-mean baseline:

| Metric | Value |
|---|---:|
| FVE | -0.005988 |
| MSE | 0.235964 |

## Result

| Metric | Value |
|---|---:|
| Best epoch | 20 |
| Best validation FVE | 0.238927 |
| Best validation RMSE | 0.422512 |
| Best validation cosine mean | 0.932064 |
| Beats train-mean baseline | yes |

## Comparison

| AR input text | Target transform | Best validation FVE | Best validation RMSE | Best validation cosine |
|---|---|---:|---:|---:|
| `reference_description` | standardize | 0.095719 | 0.460551 | 0.925471 |
| `code` | standardize | 0.238927 | 0.422512 | 0.932064 |

## Interpretation

The code-text AR run is the strongest AR result so far.

Key points:

1. Using source code as AR input improves validation FVE from `0.095719` to `0.238927`.
2. The model beats the validation train-mean baseline by a large margin.
3. Validation FVE continues improving through epoch 20, so a longer run may improve further.
4. Tokenizer truncation remains present because DistilBERT is limited to 512 tokens, but the result is still clearly better than using reference descriptions.

## Decision

Use `text_field=code`, `target_transform=standardize`, and frozen DistilBERT as the current best AR baseline.

Recommended next step:

Run a longer AR code-text training run, for example 40 epochs, or start Phase 7 AV only after deciding whether this AR is strong enough for the first NLA loop.
