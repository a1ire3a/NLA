# Phase 10c Results: Medium Qwen 0.5B Test Evaluation

## Status

Successful execution with mixed generalization.

The adapted medium 0.5B Qwen NLA loop was evaluated on the three controlled test splits. It strongly beat the mean baseline on in-domain and surface-shift tests, but did not generalize to the language-shift split.

## Setup

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| AV checkpoint | `outputs/checkpoints/qwen_av/medium_refdesc_qwen05b_lora_e3_n1000` |
| Adapted AR checkpoint | `outputs/checkpoints/qwen_nla/medium_qwen05b_generated_text_ar_adapt_e3_n1000` |
| Test activation source | Qwen2.5-Coder-1.5B layer 19 ctx512 activations |
| Batch size | 2 |
| Max generated tokens | 128 |
| Text style | generated explanation / reference-description style |

## Results

| Split | Examples | Qwen NLA FVE | Qwen NLA MSE | Mean FVE | Mean MSE | Zero FVE | Shuffled FVE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_indomain` | 500 | 0.467127 | 0.070411 | 0.000000 | 0.132134 | -11.571788 | -0.929892 |
| `test_surface_shift` | 500 | 0.490873 | 0.100553 | 0.000000 | 0.197502 | -7.253178 | -0.997270 |
| `test_language_shift` | 361 | -5.194088 | 0.118358 | 0.000000 | 0.019108 | -87.200211 | -0.945287 |

## Comparison to Debug Baseline

| Split | Debug NLA FVE | Medium Qwen NLA FVE | Change |
|---|---:|---:|---:|
| `test_indomain` | -0.758254 | 0.467127 | +1.225381 |
| `test_surface_shift` | -0.505704 | 0.490873 | +0.996577 |
| `test_language_shift` | -13.048594 | -5.194088 | +7.854506 |

The medium Qwen aligned system substantially improves over the DistilBERT/DistilGPT2 debug baseline on all three splits, even though language-shift remains below the mean baseline.

## Interpretation

The adapted medium 0.5B Qwen NLA loop shows real generalization on Python-based tests:

1. It beats the mean baseline on `test_indomain`.
2. It beats the mean baseline on `test_surface_shift`.
3. It is robust to identifier-renaming / surface-level shifts.
4. It improves substantially over the debug baseline across all test splits.

The language-shift split remains the main failure case:

1. Qwen NLA MSE is `0.118358`, which is lower than in-domain and surface-shift debug-loop MSEs, but the language-shift mean baseline MSE is extremely small: `0.019108`.
2. This makes FVE very negative.
3. The model was trained primarily on Python reference descriptions and activations, while language-shift contains C++/Java/HumanEval-X-style examples.
4. This suggests the adapted AR may be learning the generated-explanation distribution for Python-like examples rather than robust cross-language semantic invariants.

## Main Conclusion

The medium 0.5B Qwen aligned path is validated for in-domain and surface-shift evaluation. It is not yet validated for language-shift generalization.

This is the strongest result so far and supports moving toward a final 1.5B run, but the final evaluation should explicitly report language-shift as a limitation unless further cross-language training/adaptation is added.

## Decision

Proceed toward the final 1.5B run, with one caution:

- The final 1.5B run should be expected to improve in-domain and surface-shift results.
- Language-shift may remain difficult unless the training data or adaptation stage includes more language-shift-like examples.

Recommended next step:

Run the same aligned pipeline with `Qwen/Qwen2.5-Coder-1.5B-Instruct` using the established recipe:

1. Train Qwen AR on `reference_description`.
2. Train Qwen AV on `reference_description`.
3. Run validation loop before adaptation.
4. Adapt AR on AV-generated explanations.
5. Run validation loop after adaptation.
6. Evaluate adapted loop on the three controlled test splits.
