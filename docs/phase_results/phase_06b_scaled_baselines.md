# Phase 6b Results: Scaled Baselines

## Status

Successful.

Baseline evaluation completed for the scaled train and validation activation artifacts.

## Artifacts

- Train: `outputs/activations/train_qwen25_coder_15b_l19_ctx512`
- Validation: `outputs/activations/validation_qwen25_coder_15b_l19_ctx512`

## Results

| Run | Shape | Baseline | FVE | MSE |
|---|---|---|---:|---:|
| train | `(5000, 1536)` | mean | 0.000000 | 0.176965 |
| train | `(5000, 1536)` | zero | -8.362388 | 1.656818 |
| train | `(5000, 1536)` | shuffled | -1.003030 | 0.354467 |
| validation | `(500, 1536)` | mean | 0.000000 | 0.234559 |
| validation | `(500, 1536)` | zero | -6.041395 | 1.651625 |
| validation | `(500, 1536)` | shuffled | -1.052512 | 0.481436 |
| validation using train reference | `(500, 1536)` | train mean | -0.005988 | 0.235964 |

## Interpretation

The scaled baseline results are consistent.

The key AR validation reference is the validation artifact reconstructed from the train mean:

```text
FVE = -0.005988
MSE = 0.235964
```

A larger AR model should beat this reference baseline on validation.

## Decision

Proceed to Phase 6c: train AR on the scaled train and validation artifacts.

Recommended default:

```text
text_model_name_or_path = distilbert-base-uncased
text_field = reference_description
fallback_text_fields = prompt,code
target_transform = standardize
freeze_text_model = true
```
