# Phase 11 Results: Qwen 1.5B AV Reward RL

## Status

Successful.

Phase 11 adds a reward-driven AV optimization stage on top of the Phase 10d Qwen 1.5B aligned checkpoint. This is the first project phase that moves beyond supervised AV imitation toward the unsupervised / reconstruction-reward direction of the original NLA algorithm.

The reward stage optimizes AV with a reconstruction reward from AR:

```text
activation -> AV generated explanation -> AR reconstructed activation -> reward
```

The reward used in this run is:

```text
reward = -MSE(L2_normalize(reconstructed_activation), L2_normalize(original_activation))
```

## Input Checkpoint

| Field | Value |
|---|---|
| Warm-start checkpoint | `outputs/checkpoints/qwen_joint/final_qwen15b_full_e20` |
| Warm-start model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Warm-start validation FVE | 0.361623 |
| Warm-start validation MSE | 0.149737 |
| Warm-start validation cosine mean | 0.940772 |

## Main Run Setup

| Field | Value |
|---|---|
| Output directory | `outputs/checkpoints/qwen_rl/final_qwen15b_av_reward_rl` |
| Activation directory | `outputs/activations/train_qwen25_coder_15b_l19_ctx512` |
| Validation activation directory | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| Checkpoint mode | joint |
| Epochs | 3 |
| Batch size | 8 |
| Gradient accumulation steps | 2 |
| Learning rate | `5e-5` |
| Max new tokens | 64 |
| Max AR length | 256 |
| Sampling temperature | 0.7 |
| Top-p | 0.95 |
| Reward normalization | `batch_zscore` |
| KL/SFT weight | 0.01 |
| dtype | `bfloat16` |
| Train examples | 5000 |
| Validation examples | 500 |
| Activation dim | 1536 |

## Smoke Run Summary

The smoke run completed successfully at:

```text
outputs/checkpoints/qwen_rl/smoke_qwen15b_av_reward_rl
```

Smoke validation metrics:

| Metric | Value |
|---|---:|
| Validation FVE | 0.315216 |
| Validation MSE | 0.132707 |
| Validation RMSE | 0.364290 |
| Validation cosine mean | 0.950278 |
| Validation normalized MSE | 0.000064743 |
| Validation reward mean | -0.000064743 |
| Validation mean baseline MSE | 0.193794 |

The smoke run verified that the RL stage can load the Qwen 1.5B joint checkpoint, generate text, compute reward, update AV, write artifacts, and evaluate the validation loop.

## Main Run Training Metrics

| Epoch | Policy loss | SFT loss | Entropy | Train reward mean | Train normalized MSE | Train raw MSE | Mean generated length | Validation FVE | Validation normalized MSE | Validation reward mean | Best? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -1.486483 | 2.324717 | 1.421968 | -0.000061 | 0.000061 | 0.123542 | 15.2494 | 0.314614 | 0.000083 | -0.000083 | yes |
| 2 | -1.953701 | 1.802696 | 1.090397 | -0.000054 | 0.000054 | 0.102336 | 12.0096 | 0.457393 | 0.000068 | -0.000068 | yes |
| 3 | -2.183134 | 1.882315 | 1.286116 | -0.000060 | 0.000060 | 0.118435 | 13.2304 | 0.414516 | 0.000073 | -0.000073 | no |

Best epoch:

```text
2
```

## Main Run Best Validation Metrics

| Metric | Value |
|---|---:|
| Validation NLA FVE | 0.457392 |
| Validation NLA MSE | 0.127274 |
| Validation NLA RMSE | 0.356754 |
| Validation cosine mean | 0.947463 |
| Validation normalized MSE | 0.000068408 |
| Validation reward mean | -0.000068408 |
| Validation mean baseline MSE | 0.234559 |
| Validation zero baseline FVE | -6.041395 |
| Validation shuffled baseline FVE | -1.052512 |

## Comparison to Phase 10d

| Run | Validation FVE | Validation MSE | Cosine mean |
|---|---:|---:|---:|
| Phase 10d Qwen 1.5B aligned joint | 0.361623 | 0.149737 | 0.940772 |
| Phase 11 Qwen 1.5B AV reward RL | 0.457392 | 0.127274 | 0.947463 |

Phase 11 improved over the Phase 10d warm-start checkpoint:

| Metric | Improvement |
|---|---:|
| FVE gain | +0.095769 |
| MSE reduction | 0.022464 |
| Relative MSE reduction vs Phase 10d | ~15.00% |

Compared to the mean baseline:

```text
MSE improvement = 0.234559 - 0.127274 = 0.107286
Relative MSE reduction vs mean = ~45.74%
```

## Generated Explanation Audit

Validation generation summary:

| Metric | Value |
|---|---:|
| Rows | 500 |
| Mean generated chars | 43.702 |
| Min generated chars | 28 |
| Max generated chars | 697 |
| Empty outputs | 0 |
| Mean target chars | 294.772 |

Example outputs show that generated explanations are still often short and generic. Several early validation examples generated the same short phrase:

```text
Update the pipe after a move
```

This indicates that reward optimization improved reconstruction but did not yet guarantee semantically faithful natural-language explanations.

## Per-Example Validation Summary

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| Squared error | 195.492326 | 7.547575 | 3216.934570 |
| L2 error | 8.710344 | 2.747285 | 56.718025 |
| Cosine similarity | 0.947463 | 0.275497 | 0.998535 |
| Normalized MSE | 0.000068408 | 0.000001908 | 0.000943363 |

## Interpretation

Phase 11 is successful and important:

1. It implements the missing reconstruction-reward stage.
2. It improves the Qwen 1.5B validation full-loop result over Phase 10d.
3. It uses the reward direction intended for NLA-style unsupervised optimization: reconstruction quality without needing target text for the main reward.
4. It confirms that reward-driven AV optimization is feasible on the available single-GPU workflow.

However, the generated explanations remain short and sometimes generic. This suggests the system is learning activation-preserving textual codes more than fully faithful human explanations. The KL/SFT regularizer helps stabilize text, but more work would be needed to improve explanation faithfulness.

## Should We Run More Epochs?

Not immediately.

The best validation result occurred at epoch 2, while epoch 3 declined:

```text
Epoch 1 FVE: 0.314614
Epoch 2 FVE: 0.457393
Epoch 3 FVE: 0.414516
```

This suggests that simply extending the same run for more epochs is not the best next step. If more RL is run later, it should use early stopping, lower learning rate, or stronger text-quality regularization.

## Decision

Do not run more epochs of the same configuration now.

Recommended next step:

Evaluate the Phase 11 RL checkpoint on the controlled test splits:

- `test_indomain`
- `test_surface_shift`
- `test_language_shift`

This will show whether reward-driven AV optimization improves generalization beyond validation.
