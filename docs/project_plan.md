# Project Plan

This document is the active roadmap for the project.

## Current Phase

**Current phase:** Phase 10c — Qwen aligned / reconstruction-aware training.

**Immediate next step:** evaluate the adapted medium 0.5B Qwen NLA loop on the three controlled test splits.

## Completed phases

| Phase | Status |
|---|---|
| Phase 1 — Define research question and scope | Complete |
| Phase 2 — Build environment and verify model access | Complete |
| Phase 3 — Prepare datasets | Complete |
| Phase 4 — Extract and validate pilot activations | Complete |
| Phase 5 — Metrics, baselines, and AR pilot | Complete |
| Phase 6 — Scale activations and train AR on larger data | Complete as debug baseline |
| Phase 7 — Implement AV | Complete as debug baseline |
| Phase 8 — Connect full NLA loop | Complete as debug baseline |
| Phase 9 — Controlled evaluations with debug baseline | Complete |

## Phase 10 — Qwen-based aligned NLA implementation

### Phase 10a — Qwen AR/AV smoke

| Component | Model | Examples | Result |
|---|---|---:|---|
| Qwen AR smoke | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 128 train / 64 validation | Success |
| Qwen AV smoke | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 128 train / 64 validation | Success |
| Qwen AV generation smoke | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 20 validation | Success |

**Report:** `docs/phase_results/phase_10a_qwen_aligned_smoke.md`

**Status:** Complete.

### Phase 10c — Qwen aligned / reconstruction-aware training

#### Smoke run

| Run | Examples | FVE | MSE | Result |
|---|---:|---:|---:|---|
| Qwen NLA before adaptation | 32 validation | -0.077940 | 0.157551 | Success |
| Generated-text AR adaptation | 64 train / 32 validation | -0.081226 | 0.158031 | Success |
| Qwen NLA after adaptation | 32 validation | -0.081226 | 0.158031 | Success |

**Report:** `docs/phase_results/phase_10c_qwen_nla_adaptation_smoke.md`

#### Medium 0.5B run

| Run | Examples | Main result |
|---|---:|---|
| Qwen AR medium | 1000 train / 200 validation | validation FVE `0.160457` |
| Qwen AV medium | 1000 train / 200 validation | best validation loss `2.890944` |
| Qwen NLA before adaptation | 200 validation | FVE `-0.088272`, MSE `0.283378` |
| Generated-text AR adaptation | 1000 train / 200 validation | FVE `0.494062`, MSE `0.131743` |
| Qwen NLA after adaptation | 200 validation | FVE `0.494062`, MSE `0.131743` |

**Report:** `docs/phase_results/phase_10c_qwen_medium_aligned_run.md`

**Decision:** The medium 0.5B run validates the aligned Qwen direction. The next step is test-split evaluation with the adapted medium checkpoint before any final 1.5B run.

**Status:** In progress.

### Phase 10d — Final 1.5B run

**Status:** Not started.

## Phase 11 — Final controlled evaluation

**Status:** Not started.

## Phase 12 — Final report

**Status:** Not started.
