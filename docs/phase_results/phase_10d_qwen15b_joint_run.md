# Phase 10d Results: Final Qwen 1.5B Joint/Aligned Run

## Status

Successful execution.

The Phase 10d joint/aligned Qwen run completed using the full train and validation activation artifacts with `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

This run is best described as **aligned alternating supervised NLA training**, not full original NLA RL training. It trains AV with supervised text loss and trains AR on AV-generated explanations plus an anchor loss. It evaluates the complete round trip:

```text
activation -> AV generated explanation -> AR reconstruction -> metrics
```

## Setup

| Field | Value |
|---|---|
| Output directory | `outputs/checkpoints/qwen_joint/final_qwen15b_full_e20` |
| Model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Train activation artifact | `outputs/activations/train_qwen25_coder_15b_l19_ctx512` |
| Validation activation artifact | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| Target text field | `reference_description` |
| Fallback text fields | `prompt,code` |
| Target transform | `standardize` |
| Epochs | 20 |
| Batch size | 8 |
| Gradient accumulation steps | 8 |
| AV learning rate | `1e-4` |
| AR learning rate | `1e-4` |
| Max target length | 128 |
| Max AR length | 256 |
| Max new tokens | 64 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| dtype | `bfloat16` |
| Seed | 42 |
| Train limit | none |
| Validation limit | none |
| `eval_every_epoch` | false |

## Training Metrics

Because `eval_every_epoch=false`, validation metrics were only logged at the final epoch.

| Epoch | AV train loss | AR generated train MSE | AR anchor train MSE | Validation FVE | Validation MSE | Validation RMSE | Validation cosine mean | Is best |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.841930 | 1.220279 | 1.842078 | — | — | — | — | false |
| 2 | 2.447471 | 0.847466 | 1.024944 | — | — | — | — | false |
| 3 | 2.320770 | 0.993289 | 0.994444 | — | — | — | — | false |
| 4 | 2.199887 | 0.901168 | 0.972125 | — | — | — | — | false |
| 5 | 2.073427 | 0.934373 | 0.967572 | — | — | — | — | false |
| 16 | 0.570906 | 0.739365 | 0.774627 | — | — | — | — | false |
| 17 | 0.488793 | 0.721569 | 0.765858 | — | — | — | — | false |
| 18 | 0.430692 | 0.718060 | 0.734462 | — | — | — | — | false |
| 19 | 0.383456 | 0.717215 | 0.715058 | — | — | — | — | false |
| 20 | 0.344158 | 0.717384 | 0.708906 | 0.361623 | 0.149737 | 0.386959 | 0.940772 | true |

## Final Validation Result

| Method | FVE | MSE |
|---|---:|---:|
| Qwen 1.5B joint/aligned NLA | 0.361623 | 0.149737 |
| Mean baseline | 0.000000 | 0.234559 |
| Zero baseline | -6.041395 | 1.651625 |
| Shuffled baseline | -1.052512 | not logged in summary table |

The final validation result beats the mean baseline:

```text
MSE improvement over mean baseline = 0.234559 - 0.149737 = 0.084822
Relative MSE reduction = 36.16%
```

This FVE value matches the reported improvement:

```text
FVE = 0.361623
```

## Per-Example Validation Metrics

The validation per-example file contains 500 rows.

Summary:

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| squared error | 229.996442 | 6.568343 | 3643.892822 |
| L2 error | 8.707811 | 2.562878 | 60.364666 |
| cosine similarity | 0.940772 | 0.183226 | 0.998776 |

## Generated Explanation Audit

Generated explanation files were written for both train and validation.

| File | Rows | Mean generated chars | Min chars | Max chars | Empty outputs |
|---|---:|---:|---:|---:|---:|
| `train_generated_explanations.jsonl` | 5000 | 143.692 | 4 | 382 | 0 |
| `validation_generated_explanations.jsonl` | 500 | 148.242 | 14 | 378 | 0 |

Qualitative samples show that generated explanations are often short and generic, and not always semantically matched to the target reference description. Example validation rows included outputs such as:

```text
Target: We hit an unmapped region; map it into unicorn.
Generated: Add the items from the given report.

Target: Return package author and version as listed in `init.py`.
Generated: Get the default value for an argument.
```

This means reconstruction success is currently stronger than text-faithfulness. The system is learning generated text that AR can use for reconstruction, but the generated natural language is not yet consistently high-quality or semantically faithful.

## Comparison to Medium 0.5B Run

| Run | Validation FVE | Validation MSE | Notes |
|---|---:|---:|---|
| Medium Qwen 0.5B after adaptation | 0.494062 | 0.131743 | 1000 train / 200 validation, adapted AR |
| Final Qwen 1.5B joint/aligned | 0.361623 | 0.149737 | full train / full validation, 20 epochs |

The 1.5B joint run beats the mean baseline, but it does not outperform the medium 0.5B adapted run on the validation subset.

Possible reasons:

1. The 0.5B medium result was measured on a 200-example validation subset, while this 1.5B run used the full 500-example validation set.
2. `eval_every_epoch=false`, so earlier epochs may have performed better but were not recorded.
3. AV supervised loss kept improving, but generated text quality may have overfit or become less semantically faithful.
4. The joint script adapts AR to generated text but still does not directly optimize AV with reconstruction reward.

## Main Interpretation

The final 1.5B joint/aligned run is successful as a supervised/alternating NLA approximation:

1. It uses the full train and validation artifacts.
2. It completes the AV and AR coordinated training loop.
3. It beats the mean baseline on full validation.
4. It produces complete generated-text and per-example metric artifacts.

However, it does not yet solve the original unsupervised NLA objective:

- AV is not optimized by reconstruction reward.
- Generated explanations are not consistently semantically faithful.
- The result is better described as an aligned alternating supervised NLA system than as full NLA RL.

## Decision

Proceed with Phase 11 reward-driven AV optimization.

Recommended next step:

Use this checkpoint as the warm start for a reconstruction-reward AV training stage:

```text
outputs/checkpoints/qwen_joint/final_qwen15b_full_e20
```

The Phase 11 objective should optimize AV with reward from AR reconstruction quality, preferably using direction-normalized MSE as the reward signal.
