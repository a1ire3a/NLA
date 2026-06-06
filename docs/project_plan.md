# Project Plan

This document is the active roadmap for the project.

## Current Phase

**Current phase:** Phase 12 — Final report.

**Immediate next step:** prepare the final project report, quantitative tables, qualitative examples, limitations, and reproducibility commands.

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
| Phase 11 — Reward-driven AV optimization and controlled evaluation | Complete |

## Phase 10d summary

| Run | Model | Train / validation | Epochs | FVE | MSE | Mean MSE |
|---|---|---:|---:|---:|---:|---:|
| Final aligned joint NLA | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 5000 / 500 | 20 | 0.361623 | 0.149737 | 0.234559 |

Report:

- `docs/phase_results/phase_10d_qwen15b_joint_run.md`

## Phase 11 summary

### Validation

| Run | Model | Train / validation | Epochs | Best epoch | FVE | MSE | Mean MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| AV reward RL | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 5000 / 500 | 3 | 2 | 0.457392 | 0.127274 | 0.234559 |

Report:

- `docs/phase_results/phase_11_qwen15b_av_reward_rl.md`

### Controlled tests

| Split | FVE | MSE | Mean MSE | Result |
|---|---:|---:|---:|---|
| `test_indomain` | 0.400884 | 0.079164 | 0.132134 | Beats mean baseline |
| `test_surface_shift` | 0.480390 | 0.102624 | 0.197502 | Beats mean baseline |
| `test_language_shift` | -4.647290 | 0.107910 | 0.019108 | Below mean baseline |

Report:

- `docs/phase_results/phase_11_qwen15b_av_reward_rl_test_evaluation.md`

Decision: Phase 11 is complete. The reward-driven AV stage improved validation performance and generalized to in-domain and surface-shift tests. Language-shift remains the main limitation.

## Phase 12 — Final report

**Status:** Current phase.
