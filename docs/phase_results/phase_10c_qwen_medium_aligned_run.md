# Phase 10c Results: Medium Qwen 0.5B Aligned Run

## Status

Successful and scientifically important.

This medium 0.5B run provides the first strong positive result for the aligned Qwen NLA path. The Qwen NLA loop was below the mean baseline before generated-text AR adaptation, but after adaptation it achieved positive FVE and beat the mean baseline by a large margin.

## Setup

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Train activation artifact | `outputs/activations/train_qwen25_coder_15b_l19_ctx512` |
| Validation activation artifact | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| Train limit | 1000 |
| Validation limit | 200 |
| Epochs | 3 |
| Batch size | 2 |
| Learning rate | 2e-4 |
| Text style | `reference_description` / generated explanation |
| Target transform | standardize for AR |
| LoRA | enabled |

## Qwen AR Medium Training

| Epoch | Train MSE | Validation FVE | Validation MSE |
|---:|---:|---:|---:|
| 1 | 1.240579 | -0.012280 | 0.263591 |
| 2 | 0.975096 | 0.087205 | 0.237685 |
| 3 | 0.929696 | 0.160457 | 0.218611 |

Interpretation:

- Qwen AR improved consistently across epochs.
- Validation FVE became positive by epoch 2 and reached `0.160457` by epoch 3.
- This confirms that Qwen AR is viable at 0.5B scale with LoRA.

## Qwen AV Medium Training

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 3.054008 | 2.890944 |
| 2 | 2.579246 | 2.966808 |
| 3 | 2.097782 | 3.156195 |

Interpretation:

- Qwen AV trained successfully.
- Best validation loss occurred at epoch 1.
- Later epochs overfit the supervised reference-description objective.
- This does not invalidate the pipeline because the adaptation stage later produced a strong reconstruction result.

## Qwen NLA Loop Before Adaptation

| Method | FVE | MSE |
|---|---:|---:|
| Qwen NLA before adaptation | -0.088272 | 0.283378 |
| Mean baseline | 0.000000 | 0.260393 |
| Zero baseline | -5.345506 | 1.652325 |
| Shuffled baseline | -1.105155 | 0.548168 |

Interpretation:

- Before adaptation, the Qwen NLA loop still did not beat the mean baseline.
- This mirrors the earlier debug-baseline pattern: supervised AV alone does not guarantee activation-preserving generated text.

## Generated-Text AR Adaptation

| Epoch | Train MSE | Validation FVE | Validation MSE |
|---:|---:|---:|---:|
| 1 | 0.776821 | 0.490383 | 0.132701 |
| 2 | 0.738034 | 0.411105 | 0.153344 |
| 3 | 0.722674 | 0.494062 | 0.131743 |

Interpretation:

- Generated-text AR adaptation succeeded strongly.
- Validation FVE reached `0.494062`.
- Validation MSE dropped from the pre-adaptation NLA MSE of `0.283378` to `0.131743`.
- The adapted loop beats the mean baseline MSE of `0.260393`.

## Qwen NLA Loop After Adaptation

| Method | FVE | MSE |
|---|---:|---:|
| Qwen NLA after adaptation | 0.494062 | 0.131743 |
| Mean baseline | 0.000000 | 0.260393 |
| Zero baseline | -5.345506 | 1.652325 |
| Shuffled baseline | -1.105155 | 0.548168 |

## Before vs After Adaptation

| Run | FVE | MSE |
|---|---:|---:|
| Before adaptation | -0.088272 | 0.283378 |
| After adaptation | 0.494062 | 0.131743 |

This is the first clearly successful aligned NLA result in the project.

## Main Conclusion

The medium 0.5B run validates the current project direction:

1. Qwen-based AR is stronger and more aligned than the DistilBERT debug AR.
2. Qwen-based AV can generate text artifacts usable by AR.
3. Supervised AV alone is not enough.
4. Generated-text AR adaptation directly addresses the AV-to-AR mismatch.
5. The aligned Qwen NLA loop can beat the mean baseline after adaptation.

## Decision

Proceed toward a larger final run, but do not jump blindly.

Recommended next step:

- Run the adapted 0.5B Qwen NLA loop on the three controlled test splits to see whether the validation gain generalizes.

If the adapted 0.5B loop generalizes reasonably, proceed to the final 1.5B run.

If it does not generalize, inspect generated explanations and consider adjusting AV training / adaptation before spending 1.5B compute.
