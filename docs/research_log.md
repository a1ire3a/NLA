# Research Log

This file is the central narrative log for the NLA-for-code-semantics project. It records the experiment sequence, decisions, results, failures, bottlenecks, and next actions in one place. Generated tensors, model checkpoints, raw datasets, and large reports remain under ignored `outputs/` and `data/` paths.

## Project Goal

The project investigates whether a simplified Natural Language Autoencoder can recover meaningful information from residual-stream activations of a small code language model.

The working research question is:

> Can a simplified NLA recover meaningful information from residual-stream activations of a small code language model, and do the resulting explanations remain stable under surface-level and programming-language shifts?

The target domain is function-level code understanding. The target activation source is Qwen2.5-Coder residual-stream hidden states.

## Artifact Policy

Large generated artifacts are not committed to Git.

Ignored local paths include:

```text
data/raw/
data/processed/
outputs/
*.pt
*.npy
*.npz
*.safetensors
*.arrow
```

The repository should track:

- source code
- configuration files
- experiment plans
- phase reports
- this research log
- compact numerical summaries

The repository should not track:

- model checkpoints
- activation tensors
- generated JSONL reports with many examples
- raw or processed datasets

## Phase 1 — Research Question and Scope

### Objective

Turn the recruitment task into a focused, testable AI4Code interpretability project.

### Decisions

| Item | Decision |
|---|---|
| Main task | Function-level code understanding |
| Main target model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Smoke model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Activation type | residual-stream hidden state |
| Token position | final non-padding token |
| Main target layer | `19` |
| Smoke target layer | `16` |
| Primary metric | Fraction of Variance Explained, FVE |
| Initial dataset source | CodeSearchNet / CodeXGLUE-style Python functions |
| Controlled evaluation source | HumanEval-X multilingual examples |

### Status

Complete.

## Phase 2 — Environment and Model Feasibility

### Objective

Verify that the local CUDA environment can load the models, expose hidden states, extract activations, and accept `inputs_embeds` where needed for later activation-injection work.

### Script

```text
scripts/feasibility_probe.py
```

### Results

| Model | Layer | Hidden size | Layer count | Peak allocated VRAM | Result |
|---|---:|---:|---:|---:|---|
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 16 | 896 | 24 | ~0.97 GB | success |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 19 | 1536 | 28 | ~2.93 GB | success |

Both runs passed:

- tokenizer loading
- model loading
- hidden-state extraction
- final non-padding token activation selection
- `inputs_embeds` compatibility check

### Notes

The original feasibility logs included a deprecation warning about `torch_dtype`; this was later addressed in implementation prompts by preferring the newer `dtype` behavior where applicable.

### Report

```text
docs/phase_results/phase_01_feasibility_probe.md
```

## Phase 3 — Dataset Preparation

### Objective

Build reproducible processed JSONL splits for pilot, train, validation, and controlled tests.

### Script

```text
scripts/prepare_dataset.py
```

### Input Sources

- CodeSearchNet Python-style local dataset
- HumanEval-X Python, C++, and Java splits

### Output Splits

| File | Rows |
|---|---:|
| `pilot_100.jsonl` | 100 |
| `train.jsonl` | 5000 |
| `validation.jsonl` | 500 |
| `test_indomain.jsonl` | 500 |
| `test_surface_shift.jsonl` | 500 |
| `test_language_shift.jsonl` | 361 |

Total processed JSONL rows:

```text
6961
```

### Dataset Schema

Important fields include:

- `example_id`
- `source_dataset`
- `source_split`
- `split`
- `language`
- `task_family`
- `code`
- `prompt`
- `reference_description`
- `transformation_type`
- `paired_example_id`
- `metadata`

### Warnings

HumanEval-X had fewer filtered examples than the requested 500 examples for some language-specific test subsets:

- HumanEval-X Python: 33 examples available after filtering.
- HumanEval-X C++: 164 examples available after filtering.
- HumanEval-X Java: 164 examples available after filtering.

This is expected and documented as a data availability limitation, not a pipeline failure.

### Report

```text
docs/phase_results/phase_02_dataset_preparation.md
```

## Phase 4 — Pilot Activation Extraction

### Objective

Extract activation tensors for pilot examples and choose a usable context length.

### Script

```text
scripts/extract_activations.py
```

### Activation Format

Each activation artifact directory contains:

```text
activations.pt
metadata.jsonl
manifest.json
```

The selected activation is the hidden state at:

```text
selected_layer_index, final_non_padding_token
```

### Pilot Runs

| Run | Model | Layer | Max length | Examples | Activation shape | Mean | Std | Avg L2 | Truncated | Result |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| smoke10 | Qwen2.5-Coder-0.5B | 16 | 128 | 10 | `(10, 896)` | 0.013654 | 0.840399 | 25.134390 | 8 | success |
| pilot 0.5B | Qwen2.5-Coder-0.5B | 16 | 128 | 100 | `(100, 896)` | 0.021549 | 0.868290 | 25.973282 | 66 | success |
| pilot 1.5B | Qwen2.5-Coder-1.5B | 19 | 128 | 100 | `(100, 1536)` | 0.029074 | 1.275952 | 49.950233 | 66 | success |
| ctx256 | Qwen2.5-Coder-1.5B | 19 | 256 | 100 | `(100, 1536)` | 0.050653 | 1.275938 | 50.001606 | 30 | success |
| ctx512 | Qwen2.5-Coder-1.5B | 19 | 512 | 100 | `(100, 1536)` | 0.067766 | 1.287972 | 50.543163 | 1 | success |
| ctx1024 | Qwen2.5-Coder-1.5B | 19 | 1024 | 100 | `(100, 1536)` | 0.068601 | 1.288932 | 50.584785 | 0 | success |

### Decision

Use `max_length=512` for main-model extraction.

Reason:

- `ctx512` reduced pilot truncation from 66/100 to 1/100.
- `ctx1024` removed truncation entirely but is more expensive.
- `ctx512` is a practical default for the available GPU budget.

### Report

```text
docs/phase_results/phase_03_activation_extraction.md
```

## Phase 5a — Pilot Metrics and Baselines

### Objective

Validate reconstruction metrics and trivial baselines before training AR.

### Script

```text
scripts/run_evaluation.py
```

### Baselines

- mean reconstruction
- zero reconstruction
- shuffled reconstruction

### Pilot Results

| Artifact | Baseline | FVE | MSE |
|---|---|---:|---:|
| `pilot_100_qwen25_coder_15b_l19_ctx512` | mean | 0.000000 | 0.031744 |
| `pilot_100_qwen25_coder_15b_l19_ctx512` | zero | -51.401844 | 1.663463 |
| `pilot_100_qwen25_coder_15b_l19_ctx512` | shuffled | -1.014737 | 0.063957 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | mean | 0.000000 | 0.017681 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | zero | -93.230362 | 1.666052 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | shuffled | -1.001524 | 0.035388 |
| `pilot_100_qwen25_coder_05b_l16` | mean | 0.000000 | 0.452157 |
| `pilot_100_qwen25_coder_05b_l16` | zero | -0.668432 | 0.754392 |
| `pilot_100_qwen25_coder_05b_l16` | shuffled | -0.920304 | 0.868278 |

### Interpretation

The metric implementation behaved correctly:

- mean baseline FVE is 0 by construction when computed on the same tensor.
- zero and shuffled baselines are negative.
- scaled activations from 1.5B behave differently from 0.5B activations.

### Report

```text
docs/phase_results/phase_04_metrics_and_baselines.md
```

## Phase 5b — DistilBERT AR Pilot and Diagnostics

### Objective

Build the first text-to-activation reconstructor and test whether text can predict activation vectors.

This was a debug baseline, not the final architecture.

### Script

```text
scripts/train_ar.py
```

### Initial AR Baseline

| Field | Value |
|---|---|
| Activation artifact | `pilot_100_qwen25_coder_15b_l19_ctx512` |
| Text model | `distilbert-base-uncased` |
| Text field | `reference_description` |
| Encoder | frozen |
| Target transform | raw |
| Split | 80 train / 20 validation |
| Epochs | 20 |

### Initial Result

| Metric | Value |
|---|---:|
| Best epoch | 20 |
| Validation FVE | -5.110613 |
| Validation RMSE | 0.288495 |
| Validation cosine mean | 0.979651 |

### Interpretation

The AR pipeline worked, but the raw-target version was not good enough. High cosine but negative FVE suggested a scale/centering problem.

### Diagnostic Runs

| Setup | Text field | Frozen | Target transform | Best epoch | Validation FVE | Validation MSE | Validation cosine | Beats train-mean baseline? |
|---|---|---|---|---:|---:|---:|---:|---|
| refdesc center | `reference_description` | yes | center | 8 | -0.272226 | 0.017328 | 0.994898 | no |
| refdesc standardize | `reference_description` | yes | standardize | 13 | 0.056828 | 0.012846 | 0.996221 | yes |
| code center | `code` | yes | center | 8 | -0.032409 | 0.014062 | 0.995905 | yes |
| refdesc center unfrozen | `reference_description` | no | center | 10 | -0.099820 | 0.014980 | 0.995556 | no |

### Decisions

- Target normalization matters.
- Standardization is the best target transform for AR.
- Unfreezing DistilBERT on the tiny pilot did not help.
- Code text contains stronger reconstruction signal than short descriptions, but this also creates mismatch with AV-generated explanation text.

### Reports

```text
docs/phase_results/phase_05_ar_baseline.md
docs/phase_results/phase_07b_ar_diagnostics.md
```

## Phase 6a — Scaled Activation Extraction

### Objective

Move from the 100-example pilot to train/validation activation artifacts.

### Results

| Split | Examples | Shape | Mean | Std | Avg L2 | Truncated | Runtime | Result |
|---|---:|---|---:|---:|---:|---:|---:|---|
| train | 5000 | `(5000, 1536)` | 0.065013 | 1.285532 | 50.429317 | 368 | 2m24s | success |
| validation | 500 | `(500, 1536)` | 0.063659 | 1.283578 | 50.340824 | 51 | 15s | success |

### Interpretation

Train and validation activation statistics were close. `ctx512` remained usable at scale.

### Report

```text
docs/phase_results/phase_06_scaled_activation_extraction.md
```

## Phase 6b — Scaled Baselines

### Objective

Compute baselines on train and validation activation artifacts.

### Results

| Run | Baseline | FVE | MSE |
|---|---|---:|---:|
| train | mean | 0.000000 | 0.176965 |
| train | zero | -8.362388 | 1.656818 |
| train | shuffled | -1.003030 | 0.354467 |
| validation | mean | 0.000000 | 0.234559 |
| validation | zero | -6.041395 | 1.651625 |
| validation | shuffled | -1.052512 | 0.481436 |
| validation using train mean | mean | -0.005988 | 0.235964 |

### Key Reference Baseline

For AR validation, the most important baseline is validation reconstructed from the train mean:

```text
FVE = -0.005988
MSE = 0.235964
```

### Report

```text
docs/phase_results/phase_06b_scaled_baselines.md
```

## Phase 6c — Scaled DistilBERT AR

### Objective

Train AR on 5000 train examples and evaluate on the external 500-example validation artifact.

### Reference Description AR

| Field | Value |
|---|---|
| Text model | `distilbert-base-uncased` |
| Text field | `reference_description` |
| Target transform | standardize |
| Encoder | frozen |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 1e-3 |
| AR text max length | 256 |

| Metric | Value |
|---|---:|
| Best epoch | 12 |
| Validation FVE | 0.095719 |
| Validation RMSE | 0.460551 |
| Validation cosine mean | 0.925471 |
| Beats train-mean baseline | yes |

### Code AR

| Field | Value |
|---|---|
| Text model | `distilbert-base-uncased` |
| Text field | `code` |
| Target transform | standardize |
| Encoder | frozen |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 1e-3 |
| AR text max length | 512 |

| Metric | Value |
|---|---:|
| Best epoch | 20 |
| Validation FVE | 0.238927 |
| Validation RMSE | 0.422512 |
| Validation cosine mean | 0.932064 |
| Beats train-mean baseline | yes |

### Interpretation

The code-text AR was substantially stronger than the reference-description AR. However, the NLA loop requires AR input style to match AV output style. This created a later distribution mismatch, because AV generates natural-language explanations rather than raw code.

### Reports

```text
docs/phase_results/phase_06c_scaled_ar_refdesc.md
docs/phase_results/phase_06c_scaled_ar_code.md
```

## Phase 7 — DistilGPT2 AV Baseline

### Objective

Implement the first activation-to-text verbalizer.

### Script

```text
scripts/train_av.py
scripts/generate_av_explanations.py
```

### Setup

| Field | Value |
|---|---|
| AV model | `distilgpt2` |
| Activation artifact | train/validation Qwen2.5-Coder-1.5B ctx512 |
| Target text | `reference_description` |
| Epochs | 5 |
| Batch size | 8 |
| Learning rate | 5e-5 |
| Max target length | 64 |
| LM frozen | false |

### Training Results

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 3.292862 | 3.161037 |
| 2 | 2.926588 | 3.106717 |
| 3 | 2.734130 | 3.098751 |
| 4 | 2.580585 | 3.129976 |
| 5 | 2.444565 | 3.165976 |

Best validation loss occurred at epoch 3.

### Generation

- Generated validation rows during training: 500.
- Separate generation command generated 50 validation examples.

### Interpretation

The AV pipeline worked end-to-end. Mild overfitting appeared after epoch 3. This was acceptable for the debug baseline because the goal was to complete the loop, not optimize final text quality.

### Report

```text
docs/phase_results/phase_07_av_baseline.md
```

## Phase 8 — First Full NLA Loop on Validation

### Objective

Run:

```text
activation -> AV-generated explanation -> AR -> reconstructed activation -> FVE
```

### Code-Trained AR Loop

| Method | FVE | MSE |
|---|---:|---:|
| NLA loop | -0.353821 | 0.317551 |
| mean baseline | 0.000000 | 0.234559 |
| zero baseline | -6.041395 | 1.651625 |
| shuffled baseline | -1.052512 | 0.481436 |

### Reference-Description AR Loop

| Method | FVE | MSE |
|---|---:|---:|
| NLA loop | -0.285599 | 0.301549 |
| mean baseline | 0.000000 | 0.234559 |

### Interpretation

The full loop executed correctly but did not beat the mean baseline.

The reference-description AR reduced the mismatch and improved FVE from `-0.353821` to `-0.285599`, but still failed to beat the mean baseline.

Main bottleneck:

```text
AV-generated explanations were not activation-preserving enough.
```

### Report

```text
docs/phase_results/phase_08_full_nla_loop.md
```

## Phase 9 — Controlled Evaluations with Debug Baseline

### Phase 9a — Test Activation Extraction

| Split | Examples | Shape | Mean | Std | Avg L2 | Truncated | Runtime | Result |
|---|---:|---|---:|---:|---:|---:|---:|---|
| test_indomain | 500 | `(500, 1536)` | 0.066030 | 1.287167 | 50.499603 | 26 | 13s | success |
| test_surface_shift | 500 | `(500, 1536)` | 0.064251 | 1.275103 | 50.018532 | 42 | 14s | success |
| test_language_shift | 361 | `(361, 1536)` | 0.073697 | 1.296118 | 50.877804 | 0 | 3s | success |

### Phase 9b — Test Full-Loop Evaluation

| Split | NLA FVE | NLA MSE | Mean FVE | Mean MSE | Zero FVE | Shuffled FVE |
|---|---:|---:|---:|---:|---:|---:|
| test_indomain | -0.758254 | 0.232325 | 0.000000 | 0.132134 | -11.571788 | -0.929892 |
| test_surface_shift | -0.505704 | 0.297379 | 0.000000 | 0.197502 | -7.253178 | -0.997270 |
| test_language_shift | -13.048594 | 0.268444 | 0.000000 | 0.019108 | -87.200211 | -0.945287 |

### Interpretation

The loop worked on all test splits but did not beat the mean baseline.

Important details:

- NLA was better than zero on all test splits.
- NLA was better than shuffled on surface-shift and language-shift, but not in-domain.
- Language-shift FVE was extremely negative because the mean baseline MSE was very small. Raw NLA MSE was comparable to other splits, but relative to split variance it was poor.

### Reports

```text
docs/phase_results/phase_09a_test_activation_extraction.md
docs/phase_results/phase_09b_test_full_loop.md
```

## Major Debug-Baseline Conclusion

The complete pipeline was implemented:

```text
activation extraction -> metrics -> AR -> AV -> full loop -> controlled tests
```

But the DistilBERT/DistilGPT2 baseline was too weak and misaligned for final-quality NLA results.

Main limitations:

1. DistilBERT is not code-aware and not from the target model family.
2. DistilGPT2 is weak and not code-aware.
3. The best AR was trained on code, while AV generated explanation-style text.
4. The supervised AV objective imitated descriptions but did not optimize activation reconstruction.
5. The AV-generated explanations did not preserve enough activation-specific information.

Decision:

Move to Qwen-based aligned AR/AV.

## Phase 10a — Qwen-Based Aligned AR/AV Smoke Tests

### Objective

Replace the debug-baseline architecture with Qwen-family components while still using small smoke tests.

### Design

- Qwen AR: explanation text -> Qwen hidden state -> projection -> activation vector.
- Qwen AV: activation vector -> projected pseudo-token -> Qwen text generation.
- Training uses LoRA/PEFT.
- Default smoke model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- Final intended model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
- Text style is aligned around `reference_description` / explanation-style text.

### Qwen AR Smoke

| Field | Value |
|---|---:|
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Train examples | 128 |
| Validation examples | 64 |
| Base parameters | 499,809,664 |
| Trainable parameters | 5,776,896 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Epochs | 1 |
| Train MSE | 3.072038 |
| Validation FVE | -0.063300 |
| Validation MSE | 0.206061 |
| Result | success |

### Qwen AV Smoke

| Field | Value |
|---|---:|
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Train examples | 128 |
| Validation examples | 64 |
| Base parameters | 499,809,024 |
| Trainable parameters | 5,776,256 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Epochs | 1 |
| Train loss | 3.429286 |
| Validation loss | 3.137831 |
| Generated validation rows | 64 |
| Result | success |

### Qwen AV Generation Smoke

| Field | Value |
|---|---|
| Checkpoint | `outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora` |
| Activation artifact | validation ctx512 |
| Limit | 20 |
| Result | success |

### Interpretation

Phase 10a confirmed that the Qwen-based aligned implementation is operational. Metrics from a one-epoch 128-example smoke run are not final quality signals.

### Report

```text
docs/phase_results/phase_10a_qwen_aligned_smoke.md
```

## Current Phase — Phase 10c

### Objective

Implement aligned / reconstruction-aware Qwen NLA training.

### Planned Direction

The next implementation should add:

1. Qwen NLA loop script:

```text
activation -> Qwen AV -> generated explanation -> Qwen AR -> reconstructed activation -> metrics
```

2. Generated-text AR adaptation:

```text
activation -> Qwen AV-generated explanation
Qwen AV-generated explanation -> Qwen AR -> activation
```

This is not yet full differentiable joint training through discrete text. It is a practical reconstruction-aware adaptation step that addresses the current mismatch directly.

### Implementation Rules

- Use 0.5B Qwen for smoke testing.
- Do not run the final 1.5B model until joint/aligned training code is validated.
- Keep old debug baselines intact.
- Preserve artifact formats.
- Document all outcomes here and in phase-specific reports.

## Current Known Challenges

1. **Prompt truncation:** `max_length=128` was too short. `ctx512` is the practical default.
2. **Small-data pilot instability:** 100 examples were useful for debugging but not final judgment.
3. **FVE sensitivity:** Language-shift split has low variance, making FVE very negative even when raw MSE is comparable to other splits.
4. **Text distribution mismatch:** AR trained on code performs better standalone but does not match AV output style.
5. **Weak debug models:** DistilBERT/DistilGPT2 were sufficient for pipeline validation but not final-quality NLA.
6. **Supervised AV limitation:** Imitating descriptions does not guarantee activation-preserving explanations.
7. **Need for alignment:** Final NLA requires AR and AV to share the same text style and benefit from reconstruction-aware training.

## Current Best Artifacts

### Debug Baseline

| Component | Path |
|---|---|
| Best DistilBERT AR | `outputs/checkpoints/ar/train5000_val500_qwen25_coder_15b_l19_ctx512_code_distilbert_standardize` |
| DistilGPT2 AV | `outputs/checkpoints/av/train5000_val500_qwen25_coder_15b_l19_ctx512_refdesc_distilgpt2` |

### Qwen Smoke

| Component | Path |
|---|---|
| Qwen AR smoke | `outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora` |
| Qwen AV smoke | `outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora` |

### Activation Artifacts

| Split | Path |
|---|---|
| train | `outputs/activations/train_qwen25_coder_15b_l19_ctx512` |
| validation | `outputs/activations/validation_qwen25_coder_15b_l19_ctx512` |
| test_indomain | `outputs/activations/test_indomain_qwen25_coder_15b_l19_ctx512` |
| test_surface_shift | `outputs/activations/test_surface_shift_qwen25_coder_15b_l19_ctx512` |
| test_language_shift | `outputs/activations/test_language_shift_qwen25_coder_15b_l19_ctx512` |

## Next Actions

1. Implement Qwen NLA loop evaluation.
2. Implement generated-text AR adaptation.
3. Smoke test both with Qwen2.5-Coder-0.5B.
4. If smoke succeeds, run a medium 0.5B experiment.
5. Only after that, run final 1.5B aligned NLA training and controlled evaluation.
