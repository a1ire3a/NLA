# Phase 8 Results: Full NLA Loop

## Status

Implementation successful; reconstruction quality is not yet above the mean baseline.

The full pipeline ran end-to-end:

```text
activation -> AV generated explanation -> AR -> reconstructed activation -> FVE
```

## Setup

| Field | Value |
|---|---|
| Activation artifact | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| AV checkpoint | `outputs/checkpoints/av/train5000_val500_qwen25_coder_15b_l19_ctx512_refdesc_distilgpt2` |
| AR checkpoint | `outputs/checkpoints/ar/train5000_val500_qwen25_coder_15b_l19_ctx512_code_distilbert_standardize` |
| AV target during training | `reference_description` |
| AR text field during training | `code` |
| Validation examples | 500 |
| Smoke examples | 20 |
| Max generated tokens | 64 |

## Full Validation Result

| Method | FVE | MSE |
|---|---:|---:|
| NLA loop | -0.353821 | 0.317551 |
| Mean baseline | 0.000000 | 0.234559 |
| Zero baseline | -6.041395 | 1.651625 |
| Shuffled baseline | -1.052512 | 0.481436 |

## Smoke-20 Result

| Method | FVE | MSE |
|---|---:|---:|
| NLA loop | -0.344757 | 0.292151 |
| Mean baseline | 0.000000 | 0.217252 |
| Zero baseline | -6.533047 | 1.636569 |
| Shuffled baseline | -1.177026 | 0.472963 |

## Interpretation

The full NLA loop works mechanically, but the current loop does not yet beat the mean baseline.

Key observations:

1. The implementation is functional: AV generation, AR reconstruction, metric computation, and report writing all completed successfully.
2. The NLA loop is better than the zero and shuffled baselines, but worse than the mean baseline.
3. The result is consistent between the full validation run and the 20-example smoke run.
4. The main issue is likely a distribution mismatch: the AR checkpoint used here was trained with `text_field=code`, while the AV produces natural-language explanations.
5. The current AV was trained supervised against `reference_description`, not trained jointly to maximize AR reconstruction.

## Decision

Do not repeat the same full-loop run.

The next check should use the reference-description AR checkpoint with the same AV-generated explanations, because that AR was trained on text closer to the AV output distribution.

If the reference-description AR loop is also below the mean baseline, the bottleneck is likely AV text quality or the lack of joint AV-AR training.

If it improves substantially, the bottleneck in this run was mostly AR input distribution mismatch.
