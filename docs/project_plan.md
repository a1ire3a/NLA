# Project Plan

This document is the active roadmap for the project. These phase numbers are the phase numbers used from now on.

## Current Phase

**Current phase:** Phase 6c — AR training on larger data.

**Immediate next step:** run the code-text AR comparison.

## Phase 1 — Define research question and scope

**Goal:** Define the project as a compact AI4Code adaptation of Natural Language Autoencoders.

**Status:** Complete.

## Phase 2 — Build environment and verify model access

**Goal:** Validate CUDA, model loading, hidden-state extraction, and `inputs_embeds` compatibility.

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

**Status:** Complete.

## Phase 4 — Extract and validate pilot activations

**Goal:** Extract pilot activations and select a context length.

**Decision:** Use `max_length=512` for main-model extraction.

**Status:** Complete.

## Phase 5 — Metrics, baselines, and AR pilot

### Phase 5a — Pilot metrics and baselines

**Status:** Complete.

### Phase 5b — AR pilot and diagnostics

| Setup | Text field | Target transform | Validation FVE | Beats train-mean baseline? |
|---|---|---|---:|---|
| refdesc raw | `reference_description` | raw | -5.110613 | no |
| refdesc center | `reference_description` | center | -0.272226 | no |
| refdesc standardize | `reference_description` | standardize | 0.056828 | yes |
| code center | `code` | center | -0.032409 | yes |

**Decision:** Use target standardization for AR training.

**Status:** Complete for pilot.

## Phase 6 — Scale activations and train AR on larger data

### Phase 6a — Scaled activation extraction

| Artifact | Examples | Shape | Truncated | Status |
|---|---:|---|---:|---|
| `train_qwen25_coder_15b_l19_ctx512` | 5000 | `(5000, 1536)` | 368 | Complete |
| `validation_qwen25_coder_15b_l19_ctx512` | 500 | `(500, 1536)` | 51 | Complete |

**Report:** `docs/phase_results/phase_06_scaled_activation_extraction.md`

### Phase 6b — Scaled baselines

| Run | Baseline | FVE | MSE |
|---|---|---:|---:|
| train | mean | 0.000000 | 0.176965 |
| train | zero | -8.362388 | 1.656818 |
| train | shuffled | -1.003030 | 0.354467 |
| validation | mean | 0.000000 | 0.234559 |
| validation | zero | -6.041395 | 1.651625 |
| validation | shuffled | -1.052512 | 0.481436 |
| validation using train reference | train mean | -0.005988 | 0.235964 |

**Report:** `docs/phase_results/phase_06b_scaled_baselines.md`

**Status:** Complete.

### Phase 6c — AR training on larger data

| Setup | Text field | Target transform | Best epoch | Validation FVE | Beats train-mean baseline? |
|---|---|---|---:|---:|---|
| refdesc DistilBERT frozen | `reference_description` | standardize | 12 | 0.095719 | yes |

**Report:** `docs/phase_results/phase_06c_scaled_ar_refdesc.md`

**Status:** In progress. Next run: code-text comparison.

## Phase 7 — Implement AV

**Status:** Not started.

## Phase 8 — Connect the full NLA loop

**Status:** Not started.

## Phase 9 — Controlled evaluations

**Status:** Not started.

## Phase 10 — Final report

**Status:** Not started.
