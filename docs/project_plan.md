# Project Plan

This document is the active roadmap for the project. These phase numbers are the phase numbers used from now on.

## Current Phase

**Current phase:** Phase 6 — Scale activations and train AR on larger data.

**Immediate next step:** run baseline evaluation for the train and validation activation artifacts.

## Phase 1 — Define research question and scope

**Goal:** Define the project as a compact AI4Code adaptation of Natural Language Autoencoders.

**Status:** Complete.

## Phase 2 — Build environment and verify model access

**Goal:** Validate CUDA, model loading, hidden-state extraction, and `inputs_embeds` compatibility.

**Results:**

| Model | Layer | Hidden size | Result |
|---|---:|---:|---|
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 16 | 896 | Success |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 19 | 1536 | Success |

**Report:** `docs/phase_results/phase_01_feasibility_probe.md`

**Status:** Complete.

## Phase 3 — Prepare datasets

**Goal:** Convert raw datasets into project-standard JSONL files.

| File | Rows |
|---|---:|
| `pilot_100.jsonl` | 100 |
| `train.jsonl` | 5000 |
| `validation.jsonl` | 500 |
| `test_indomain.jsonl` | 500 |
| `test_surface_shift.jsonl` | 500 |
| `test_language_shift.jsonl` | 361 |

**Report:** `docs/phase_results/phase_02_dataset_preparation.md`

**Status:** Complete.

## Phase 4 — Extract and validate pilot activations

**Goal:** Extract pilot activations and select a context length.

| Artifact | Model | Layer | Max length | Examples | Shape | Truncated |
|---|---|---:|---:|---:|---|---:|
| `pilot_100_qwen25_coder_05b_l16` | 0.5B | 16 | 128 | 100 | `(100, 896)` | 66 |
| `pilot_100_qwen25_coder_15b_l19_ctx512` | 1.5B | 19 | 512 | 100 | `(100, 1536)` | 1 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | 1.5B | 19 | 1024 | 100 | `(100, 1536)` | 0 |

**Decision:** Use `max_length=512` for main-model extraction.

**Report:** `docs/phase_results/phase_03_activation_extraction.md`

**Status:** Complete.

## Phase 5 — Metrics, baselines, and AR pilot

### Phase 5a — Metrics and baselines

| Artifact | Mean FVE | Zero FVE | Shuffled FVE |
|---|---:|---:|---:|
| `pilot_100_qwen25_coder_15b_l19_ctx512` | 0.000000 | -51.401844 | -1.014737 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | 0.000000 | -93.230362 | -1.001524 |
| `pilot_100_qwen25_coder_05b_l16` | 0.000000 | -0.668432 | -0.920304 |

**Report:** `docs/phase_results/phase_04_metrics_and_baselines.md`

**Status:** Complete.

### Phase 5b — AR pilot and diagnostics

| Setup | Text field | Target transform | Validation FVE | Beats train-mean baseline? |
|---|---|---|---:|---|
| refdesc raw | `reference_description` | raw | -5.110613 | no |
| refdesc center | `reference_description` | center | -0.272226 | no |
| refdesc standardize | `reference_description` | standardize | 0.056828 | yes |
| code center | `code` | center | -0.032409 | yes |

**Decision:** Use target standardization for AR training. Best pilot setting: `reference_description + standardize + frozen DistilBERT`.

**Reports:**

- `docs/phase_results/phase_05_ar_baseline.md`
- `docs/phase_results/phase_07b_ar_diagnostics.md`

**Status:** Complete for pilot.

## Phase 6 — Scale activations and train AR on larger data

### Phase 6a — Scaled activation extraction

| Artifact | Examples | Shape | Truncated | Status |
|---|---:|---|---:|---|
| `train_qwen25_coder_15b_l19_ctx512` | 5000 | `(5000, 1536)` | 368 | Complete |
| `validation_qwen25_coder_15b_l19_ctx512` | 500 | `(500, 1536)` | 51 | Complete |

**Report:** `docs/phase_results/phase_06_scaled_activation_extraction.md`

### Phase 6b — Baselines on train and validation artifacts

**Status:** Next step.

### Phase 6c — AR training on larger data

**Planned default:** `distilbert-base-uncased`, `reference_description`, `standardize`, frozen text encoder.

**Status:** Not started.

## Phase 7 — Implement AV

**Status:** Not started.

## Phase 8 — Connect the full NLA loop

**Status:** Not started.

## Phase 9 — Controlled evaluations

**Status:** Not started.

## Phase 10 — Final report

**Status:** Not started.
