# Project Plan

This document is the active roadmap for the project.

## Current Phase

**Current phase:** Phase 11 — Reward-driven AV optimization.

**Immediate next step:** use the Phase 10d Qwen 1.5B checkpoint as a warm start for reconstruction-reward AV training.

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
| Phase 10d — Final 1.5B aligned Qwen run | Complete |

## Phase 10c summary

| Run | FVE | MSE |
|---|---:|---:|
| Medium Qwen 0.5B before adaptation | -0.088272 | 0.283378 |
| Medium Qwen 0.5B after adaptation | 0.494062 | 0.131743 |

Reports:

- `docs/phase_results/phase_10c_qwen_medium_aligned_run.md`
- `docs/phase_results/phase_10c_qwen_medium_test_evaluation.md`

## Phase 10d summary

| Run | Model | Train / validation | Epochs | FVE | MSE | Mean MSE |
|---|---|---:|---:|---:|---:|---:|
| Final aligned joint NLA | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 5000 / 500 | 20 | 0.361623 | 0.149737 | 0.234559 |

Report:

- `docs/phase_results/phase_10d_qwen15b_joint_run.md`

Decision: the final 1.5B aligned run beats the mean baseline on full validation, but it is still an aligned supervised/alternating approximation. Proceed to Phase 11 to add reward-driven AV optimization.

## Phase 11 — Reward-driven AV optimization

**Status:** Current phase.

## Phase 12 — Final report

**Status:** Not started.
