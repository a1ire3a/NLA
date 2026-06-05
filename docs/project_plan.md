# Project Plan

This document is the active roadmap for the project. These phase numbers are the phase numbers used from now on.

## Current Phase

**Current phase:** Phase 10 — Qwen-based aligned NLA implementation.

**Immediate next step:** replace the DistilBERT/DistilGPT2 debug baselines with Qwen-based AR and AV components, using the 0.5B model for smoke tests and reserving the 1.5B model for the final run.

## Phase 1 — Define research question and scope

**Status:** Complete.

## Phase 2 — Build environment and verify model access

**Status:** Complete.

## Phase 3 — Prepare datasets

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

**Decision:** Use `max_length=512` for main-model extraction.

**Status:** Complete.

## Phase 5 — Metrics, baselines, and AR pilot

**Status:** Complete.

## Phase 6 — Scale activations and train AR on larger data

| Component | Best result |
|---|---|
| AR on `reference_description` | validation FVE `0.095719` |
| AR on `code` | validation FVE `0.238927` |

**Status:** Complete as a debug baseline.

## Phase 7 — Implement AV

| Setup | Target field | LM | Epochs | Best validation loss | Status |
|---|---|---|---:|---:|---|
| supervised AV baseline | `reference_description` | `distilgpt2` | 5 | 3.098751 | Complete |

**Status:** Complete as a debug baseline.

## Phase 8 — Connect the full NLA loop

| Split | NLA FVE | NLA MSE | Mean MSE |
|---|---:|---:|---:|
| validation | -0.353821 | 0.317551 | 0.234559 |

**Report:** `docs/phase_results/phase_08_full_nla_loop.md`

**Status:** Complete as a debug baseline.

## Phase 9 — Controlled evaluations with the debug baseline

| Split | NLA FVE | NLA MSE | Mean MSE | Shuffled FVE |
|---|---:|---:|---:|---:|
| `test_indomain` | -0.758254 | 0.232325 | 0.132134 | -0.929892 |
| `test_surface_shift` | -0.505704 | 0.297379 | 0.197502 | -0.997270 |
| `test_language_shift` | -13.048594 | 0.268444 | 0.019108 | -0.945287 |

**Reports:**

- `docs/phase_results/phase_09a_test_activation_extraction.md`
- `docs/phase_results/phase_09b_test_full_loop.md`

**Status:** Complete for the debug baseline.

## Phase 10 — Qwen-based aligned NLA implementation

**Goal:** Move from debug baselines to a stronger, aligned NLA implementation.

**Problem with the debug baseline:**

- AR used DistilBERT, which is not code-aware and not from the target-model family.
- AV used DistilGPT2, which is also weak and not code-aware.
- AR and AV were not aligned: the best AR was trained on `code`, while AV generated natural-language explanations.

**Implementation direction:**

- Use Qwen-based AR and AV components.
- Use LoRA/PEFT rather than full fine-tuning.
- Keep AR input style aligned with AV output style.
- Use 0.5B Qwen for implementation smoke tests.
- Use 1.5B Qwen only for the final serious run.

### Phase 10a — Implement Qwen-based aligned AR/AV

**Goal:** Add Qwen-based model classes and training scripts while preserving the existing debug baselines.

**Default smoke model:** `Qwen/Qwen2.5-Coder-0.5B-Instruct`

**Final model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`

**Status:** Current step.

### Phase 10b — Smoke test Qwen AR and Qwen AV separately

**Goal:** Verify that Qwen-based AR and AV train and generate on a tiny subset without breaking memory or artifact formats.

**Status:** Not started.

### Phase 10c — Implement aligned / joint training

**Goal:** Add reconstruction-aware training where AV-generated text is evaluated through AR reconstruction.

**Status:** Not started.

### Phase 10d — Final 1.5B run

**Goal:** Run the aligned Qwen-based NLA on the main train/validation/test artifacts.

**Status:** Not started.

## Phase 11 — Final controlled evaluation

**Goal:** Evaluate the final Qwen-based aligned NLA on in-domain, surface-shift, and language-shift splits.

**Status:** Not started.

## Phase 12 — Final report

**Goal:** Produce final README/report, quantitative tables, qualitative examples, limitations, and reproducibility commands.

**Status:** Not started.
