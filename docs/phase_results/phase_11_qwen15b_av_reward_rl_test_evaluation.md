# Phase 11 Results: Qwen 1.5B AV Reward RL Test Evaluation

## Status

Successful execution with mixed generalization.

The Phase 11 reward-trained AV checkpoint was evaluated on the three controlled test splits using the AR component from the Phase 10d joint checkpoint.

Evaluation path:

```text
activation -> RL-optimized AV generated explanation -> Phase 10d AR reconstruction -> metrics
```

## Setup

| Field | Value |
|---|---|
| RL AV checkpoint | `outputs/checkpoints/qwen_rl/final_qwen15b_av_reward_rl` |
| AR source checkpoint | `outputs/checkpoints/qwen_joint/final_qwen15b_full_e20` |
| Model family | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Activation source | Qwen2.5-Coder-1.5B layer 19 ctx512 activations |
| Batch size | 2 |
| Max generated tokens | 64 |
| Seed | 42 |

## Results

| Split | Examples | Qwen NLA FVE | Qwen NLA MSE | Mean FVE | Mean MSE | Zero FVE | Shuffled FVE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_indomain` | 500 | 0.400884 | 0.079164 | 0.000000 | 0.132134 | -11.571788 | -0.929892 |
| `test_surface_shift` | 500 | 0.480390 | 0.102624 | 0.000000 | 0.197502 | -7.253178 | -0.997270 |
| `test_language_shift` | 361 | -4.647290 | 0.107910 | 0.000000 | 0.019108 | -87.200211 | -0.945287 |

## Interpretation

The Phase 11 RL checkpoint generalizes well to Python-based test settings:

1. It beats the mean baseline on `test_indomain`.
2. It beats the mean baseline on `test_surface_shift`.
3. It remains robust to surface-level identifier-renaming shifts.

The language-shift split remains the main failure case:

1. The raw MSE is `0.107910`, which is lower than the Phase 10c medium Qwen language-shift MSE of `0.118358`.
2. The FVE is still very negative because the language-shift mean baseline MSE is extremely small: `0.019108`.
3. This indicates that cross-language generalization remains unsolved under the current Python-heavy training/adaptation setup.

## Comparison to Earlier Systems

### Compared to the DistilBERT/DistilGPT2 debug baseline

| Split | Debug NLA FVE | Phase 11 RL FVE | Change |
|---|---:|---:|---:|
| `test_indomain` | -0.758254 | 0.400884 | +1.159138 |
| `test_surface_shift` | -0.505704 | 0.480390 | +0.986094 |
| `test_language_shift` | -13.048594 | -4.647290 | +8.401304 |

### Compared to the medium Qwen 0.5B adapted system

| Split | Medium Qwen 0.5B FVE | Phase 11 Qwen 1.5B RL FVE | Change |
|---|---:|---:|---:|
| `test_indomain` | 0.467127 | 0.400884 | -0.066243 |
| `test_surface_shift` | 0.490873 | 0.480390 | -0.010483 |
| `test_language_shift` | -5.194088 | -4.647290 | +0.546798 |

The Phase 11 1.5B RL system is slightly below the medium 0.5B adapted system on in-domain and surface-shift FVE, but it improves language-shift raw/FVE performance relative to that system. The comparison is not perfectly controlled because the 0.5B and 1.5B systems used different training recipes and checkpoints.

## Main Conclusion

Phase 11 successfully demonstrates the missing reconstruction-reward stage:

```text
AV is optimized with reconstruction reward from AR, not only supervised CE on reference descriptions.
```

The resulting system:

- improves validation performance over Phase 10d,
- beats the mean baseline on in-domain and surface-shift tests,
- improves over the original debug baseline across all test splits,
- but does not solve language-shift generalization.

## Decision

This is sufficient for the final project report as a resource-constrained NLA-style system with a reward-driven AV optimization stage.

Recommended final reporting stance:

1. Present Phase 11 as a compact, single-GPU approximation of the original NLA RL stage.
2. Report in-domain and surface-shift as successful controlled generalization settings.
3. Report language-shift as the main limitation.
4. Do not run more RL epochs with the same configuration; the validation curve already showed epoch 2 as best and epoch 3 as worse.

Next step:

Prepare the final report and qualitative analysis.
