# Project Plan

This document is the active roadmap for the project.

## Current Phase

**Current phase:** Phase 11 — Controlled evaluation of reward-driven AV optimization.

**Immediate next step:** evaluate the Phase 11 RL checkpoint on `test_indomain`, `test_surface_shift`, and `test_language_shift`.

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
| Phase 11 — Reward-driven AV optimization | Complete on validation |

## Phase 10d summary

| Run | Model | Train / validation | Epochs | FVE | MSE | Mean MSE |
|---|---|---:|---:|---:|---:|---:|
| Final aligned joint NLA | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 5000 / 500 | 20 | 0.361623 | 0.149737 | 0.234559 |

Report:

- `docs/phase_results/phase_10d_qwen15b_joint_run.md`

## Phase 11 summary

| Run | Model | Train / validation | Epochs | Best epoch | FVE | MSE | Mean MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| AV reward RL | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 5000 / 500 | 3 | 2 | 0.457392 | 0.127274 | 0.234559 |

Report:

- `docs/phase_results/phase_11_qwen15b_av_reward_rl.md`

Decision: do not run more epochs with the same configuration now. The best validation result occurred at epoch 2 and epoch 3 declined. Evaluate the RL checkpoint on controlled test splits next.

## Phase 12 — Final report

**Status:** Not started.
