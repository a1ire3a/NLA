# Project Plan

This document is the active roadmap for the project. These phase numbers are the phase numbers used from now on.

## Current Phase

**Current phase:** Phase 10c — Qwen aligned / reconstruction-aware training.

**Immediate next step:** run a medium 0.5B Qwen aligned experiment before the final 1.5B run.

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

**Status:** Complete as a debug baseline.

## Phase 9 — Controlled evaluations with the debug baseline

| Split | NLA FVE | NLA MSE | Mean MSE | Shuffled FVE |
|---|---:|---:|---:|---:|
| `test_indomain` | -0.758254 | 0.232325 | 0.132134 | -0.929892 |
| `test_surface_shift` | -0.505704 | 0.297379 | 0.197502 | -0.997270 |
| `test_language_shift` | -13.048594 | 0.268444 | 0.019108 | -0.945287 |

**Status:** Complete for the debug baseline.

## Phase 10 — Qwen-based aligned NLA implementation

**Goal:** Move from debug baselines to a stronger, aligned NLA implementation.

### Phase 10a — Implement Qwen-based aligned AR/AV

| Component | Model | Examples | Result |
|---|---|---:|---|
| Qwen AR smoke | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 128 train / 64 validation | Success |
| Qwen AV smoke | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 128 train / 64 validation | Success |
| Qwen AV generation smoke | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 20 validation | Success |

**Report:** `docs/phase_results/phase_10a_qwen_aligned_smoke.md`

**Status:** Complete.

### Phase 10b — Smoke test Qwen AR and Qwen AV separately

**Status:** Complete as part of Phase 10a smoke testing.

### Phase 10c — Qwen aligned / reconstruction-aware training

| Run | Examples | FVE | MSE | Status |
|---|---:|---:|---:|---|
| Qwen NLA before adaptation | 32 validation | -0.077940 | 0.157551 | Success |
| Generated-text AR adaptation | 64 train / 32 validation | -0.081226 | 0.158031 | Success |
| Qwen NLA after adaptation | 32 validation | -0.081226 | 0.158031 | Success |

**Report:** `docs/phase_results/phase_10c_qwen_nla_adaptation_smoke.md`

**Interpretation:** The implementation works, but the tiny one-epoch generated-text AR adaptation did not improve reconstruction. This smoke run validates code paths, not final quality.

**Next step:** run a medium 0.5B Qwen aligned experiment with more examples and epochs.

**Status:** In progress.

### Phase 10d — Final 1.5B run

**Goal:** Run the aligned Qwen-based NLA on the main train/validation/test artifacts.

**Status:** Not started.

## Phase 11 — Final controlled evaluation

**Goal:** Evaluate the final Qwen-based aligned NLA on in-domain, surface-shift, and language-shift splits.

**Status:** Not started.

## Phase 12 — Final report

**Goal:** Produce final README/report, quantitative tables, qualitative examples, limitations, and reproducibility commands.

**Status:** Not started.
