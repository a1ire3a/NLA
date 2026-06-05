# Project Plan

This document is the active roadmap for the project.

## Current Phase

**Current phase:** Phase 10d — Final 1.5B aligned Qwen run.

**Immediate next step:** run the established aligned Qwen recipe with `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

## Completed phases

| Phase | Status |
|---|---|
| Phase 1 — Define research question and scope | Complete |
| Phase 2 — Build environment and verify model access | Complete |
| Phase 3 — Prepare datasets | Complete |
| Phase 4 — Extract and validate pilot activations | Complete |
| Phase 5 — Metrics, baselines, and AR pilot | Complete |
| Phase 6 — Scaled debug AR | Complete |
| Phase 7 — Debug AV | Complete |
| Phase 8 — Debug full loop | Complete |
| Phase 9 — Debug controlled tests | Complete |
| Phase 10a — Qwen AR/AV smoke | Complete |
| Phase 10c — Medium 0.5B aligned Qwen run | Complete |

## Phase 10c summary

### Medium validation result

| Run | FVE | MSE |
|---|---:|---:|
| Qwen NLA before adaptation | -0.088272 | 0.283378 |
| Qwen NLA after adaptation | 0.494062 | 0.131743 |

### Medium test result

| Split | FVE | MSE | Mean MSE |
|---|---:|---:|---:|
| `test_indomain` | 0.467127 | 0.070411 | 0.132134 |
| `test_surface_shift` | 0.490873 | 0.100553 | 0.197502 |
| `test_language_shift` | -5.194088 | 0.118358 | 0.019108 |

Reports:

- `docs/phase_results/phase_10c_qwen_medium_aligned_run.md`
- `docs/phase_results/phase_10c_qwen_medium_test_evaluation.md`

Decision: proceed to the final 1.5B aligned Qwen run. The language-shift split remains a limitation to report separately.

## Phase 10d — Final 1.5B run

**Status:** Current phase.

## Phase 11 — Final controlled evaluation

**Status:** Not started.

## Phase 12 — Final report

**Status:** Not started.
