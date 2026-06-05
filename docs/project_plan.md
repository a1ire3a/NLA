# Project Plan

This document is the active roadmap for the project. These phase numbers are the phase numbers used from now on.

## Current Phase

**Current phase:** Phase 10 — Improve NLA reconstruction.

**Immediate next step:** implement reconstruction-driven or generated-text-aware improvement instead of repeating the same supervised AV baseline.

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

**Status:** Complete enough for first NLA implementation.

## Phase 7 — Implement AV

| Setup | Target field | LM | Epochs | Best validation loss | Status |
|---|---|---|---:|---:|---|
| supervised AV baseline | `reference_description` | `distilgpt2` | 5 | 3.098751 | Complete |

**Status:** Complete enough for first NLA implementation.

## Phase 8 — Connect the full NLA loop

### Validation full-loop result

| Method | FVE | MSE |
|---|---:|---:|
| NLA loop | -0.353821 | 0.317551 |
| mean baseline | 0.000000 | 0.234559 |

**Report:** `docs/phase_results/phase_08_full_nla_loop.md`

**Status:** Complete for validation baseline.

## Phase 9 — Controlled evaluations

### Phase 9a — Test activation extraction

| Split | Examples | Shape | Truncated | Status |
|---|---:|---|---:|---|
| `test_indomain` | 500 | `(500, 1536)` | 26 | Complete |
| `test_surface_shift` | 500 | `(500, 1536)` | 42 | Complete |
| `test_language_shift` | 361 | `(361, 1536)` | 0 | Complete |

**Report:** `docs/phase_results/phase_09a_test_activation_extraction.md`

### Phase 9b — Test full-loop evaluation

| Split | NLA FVE | NLA MSE | Mean MSE | Shuffled FVE |
|---|---:|---:|---:|---:|
| `test_indomain` | -0.758254 | 0.232325 | 0.132134 | -0.929892 |
| `test_surface_shift` | -0.505704 | 0.297379 | 0.197502 | -0.997270 |
| `test_language_shift` | -13.048594 | 0.268444 | 0.019108 | -0.945287 |

**Report:** `docs/phase_results/phase_09b_test_full_loop.md`

**Status:** Complete for current supervised baseline.

## Phase 10 — Improve NLA reconstruction

**Goal:** Move beyond the supervised AV baseline toward reconstruction-aware training.

**Current bottleneck:** AV-generated explanations do not preserve enough activation-specific information for AR reconstruction.

**Candidate next steps:**

1. Train AR on AV-generated explanations as an adaptation step.
2. Add reconstruction-driven AV fine-tuning.
3. Move AV/AR to stronger or same-family code-aware models.
4. Add qualitative analysis of AV-generated explanations before choosing the expensive path.

**Status:** Current phase.
