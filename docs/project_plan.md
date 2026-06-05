# Project Plan

This document is the active roadmap for the project. These phase numbers are the phase numbers used from now on.

## Current Phase

**Current phase:** Phase 8 — Connect the full NLA loop.

**Immediate next step:** run the full NLA loop again with the reference-description AR checkpoint to isolate AR/AV text-distribution mismatch.

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

**Status:** Complete.

## Phase 6 — Scale activations and train AR on larger data

### Phase 6a — Scaled activation extraction

| Artifact | Examples | Shape | Truncated | Status |
|---|---:|---|---:|---|
| `train_qwen25_coder_15b_l19_ctx512` | 5000 | `(5000, 1536)` | 368 | Complete |
| `validation_qwen25_coder_15b_l19_ctx512` | 500 | `(500, 1536)` | 51 | Complete |

### Phase 6b — Scaled baselines

| Run | Baseline | FVE | MSE |
|---|---|---:|---:|
| train | mean | 0.000000 | 0.176965 |
| validation using train reference | train mean | -0.005988 | 0.235964 |

### Phase 6c — AR training on larger data

| Setup | Text field | Target transform | Best epoch | Validation FVE | Validation RMSE | Beats train-mean baseline? |
|---|---|---|---:|---:|---:|---|
| refdesc DistilBERT frozen | `reference_description` | standardize | 12 | 0.095719 | 0.460551 | yes |
| code DistilBERT frozen | `code` | standardize | 20 | 0.238927 | 0.422512 | yes |

**Decision:** The current AR baseline is sufficient for proceeding to the first AV implementation.

**Status:** Complete enough for first NLA implementation.

## Phase 7 — Implement AV

| Setup | Target field | LM | Epochs | Best validation loss | Status |
|---|---|---|---:|---:|---|
| supervised AV baseline | `reference_description` | `distilgpt2` | 5 | 3.098751 | Complete |

**Report:** `docs/phase_results/phase_07_av_baseline.md`

**Status:** Complete enough for first NLA implementation.

## Phase 8 — Connect the full NLA loop

**Goal:** Run:

```text
activation -> AV generated explanation -> AR -> reconstructed activation -> FVE
```

### Phase 8a — AV-to-code-AR loop

| Method | FVE | MSE |
|---|---:|---:|
| NLA loop | -0.353821 | 0.317551 |
| mean baseline | 0.000000 | 0.234559 |
| zero baseline | -6.041395 | 1.651625 |
| shuffled baseline | -1.052512 | 0.481436 |

**Report:** `docs/phase_results/phase_08_full_nla_loop.md`

**Interpretation:** The loop runs end-to-end but does not beat the mean baseline. The likely issue is text-distribution mismatch: the AR checkpoint used here was trained on code text, while the AV produces natural-language explanations.

**Next diagnostic:** Run the same loop using the reference-description AR checkpoint.

**Status:** In progress.

## Phase 9 — Controlled evaluations

**Status:** Not started.

## Phase 10 — Final report

**Status:** Not started.
