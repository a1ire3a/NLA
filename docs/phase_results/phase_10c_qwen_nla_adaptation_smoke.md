# Phase 10c Results: Qwen NLA Loop and Generated-Text AR Adaptation Smoke

## Status

Successful execution; quality is not expected to be final because this is a tiny smoke run.

This phase verified that the Qwen-based aligned NLA loop and generated-text AR adaptation pipeline run end-to-end with the 0.5B Qwen smoke model.

## Setup

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Validation activation artifact | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| Qwen AV checkpoint | `outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora` |
| Qwen AR checkpoint before adaptation | `outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora` |
| Adapted Qwen AR checkpoint | `outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt` |
| Limit for loop smoke | 32 validation examples |
| Adaptation train limit | 64 examples |
| Adaptation validation limit | 32 examples |
| Epochs | 1 |
| Batch size | 2 |
| Learning rate | 2e-4 |
| Max generated tokens | 128 |
| Max AR length | 256 |

## Qwen NLA Loop Before Adaptation

Command summary:

```text
run_qwen_nla_loop.py with 32 validation examples, Qwen AV smoke checkpoint, Qwen AR smoke checkpoint
```

Results:

| Method | FVE | MSE |
|---|---:|---:|
| Qwen NLA | -0.077940 | 0.157551 |
| Mean baseline | 0.000000 | 0.146159 |
| Zero baseline | -10.306170 | 1.652502 |
| Shuffled baseline | -1.105089 | 0.307678 |

Interpretation:

- The Qwen full-loop execution works.
- It is close to but still below the mean baseline on this tiny smoke subset.
- It is substantially better than zero and shuffled baselines.

## Generated-Text AR Adaptation

Command summary:

```text
train_qwen_nla_reconstruction.py with 64 train examples, 32 validation examples, 1 epoch
```

Results:

| Metric | Value |
|---|---:|
| Generated train rows | 64 |
| Generated validation rows | 32 |
| Train MSE | 0.778261 |
| Validation FVE | -0.081226 |
| Validation MSE | 0.158031 |

Artifacts written:

```text
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/projection_head.pt
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/tokenizer/
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/qwen_adapter/
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/model.pt
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/train_generated_explanations.jsonl
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/validation_generated_explanations.jsonl
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/training_metrics.csv
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/validation_predictions.pt
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/validation_targets.pt
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/validation_metadata.jsonl
outputs/checkpoints/qwen_nla/smoke_qwen05b_generated_text_ar_adapt/train_qwen_nla_reconstruction_manifest.json
```

## Qwen NLA Loop After Adaptation

Command summary:

```text
run_qwen_nla_loop.py with 32 validation examples, Qwen AV smoke checkpoint, adapted Qwen AR checkpoint
```

Results:

| Method | FVE | MSE |
|---|---:|---:|
| Qwen NLA after adaptation | -0.081226 | 0.158031 |
| Mean baseline | 0.000000 | 0.146159 |
| Zero baseline | -10.306170 | 1.652502 |
| Shuffled baseline | -1.105089 | 0.307678 |

## Before vs After Adaptation

| Run | FVE | MSE |
|---|---:|---:|
| Before generated-text AR adaptation | -0.077940 | 0.157551 |
| After generated-text AR adaptation | -0.081226 | 0.158031 |

## Interpretation

The Phase 10c smoke test is successful as an implementation check, but the 1-epoch 64-example adaptation did not improve reconstruction quality.

This is not a failure of the project direction because the run was intentionally tiny. It confirms that:

1. Qwen AV generation can feed Qwen AR.
2. The Qwen full-loop script works.
3. Generated-text AR adaptation runs and writes complete artifacts.
4. The before/after loop comparison works.

The next step should not be to move directly to the 1.5B final run. The next step should be a medium 0.5B experiment with more examples and more epochs to verify whether generated-text adaptation improves when given enough data.

## Decision

Proceed to a medium 0.5B Qwen aligned experiment:

- train Qwen AR on more examples with `reference_description`
- train Qwen AV on more examples with `reference_description`
- run Qwen NLA loop before adaptation
- adapt AR on AV-generated explanations with more examples and epochs
- run Qwen NLA loop after adaptation

Only after this medium 0.5B run should the project move to the 1.5B final run.
