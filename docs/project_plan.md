# Project Plan

This document is the compact roadmap for the whole project. The README is the final short report; this file explains the sequence of phases and points to detailed phase reports.

## Current status

**Current phase:** Phase 12 — Final report and submission.

**Submission-ready state:** yes. The code, README, phase reports, experiment registries, and reproduction instructions are now in the repository.

Main entry points:

- [README.md](../README.md) — final short report.
- [docs/research_log.md](research_log.md) — narrative record of decisions, results, and issues.
- [docs/phase_results/](phase_results/) — detailed phase-by-phase reports.
- [experiments/](../experiments/) — lightweight CSV experiment registries.
- [docs/manual_installation.md](manual_installation.md) — setup and reproduction guide.

## Phase timeline

| Phase | Title | Status | Key output | Detailed report |
|---|---|---|---|---|
| 1 | Feasibility probe | Complete | Qwen 0.5B and 1.5B could be loaded on CUDA; selected-layer activations and `inputs_embeds` path worked. | [phase_01_feasibility_probe.md](phase_results/phase_01_feasibility_probe.md) |
| 2 | Dataset preparation | Complete | Built 6961 processed examples: pilot, train, validation, in-domain test, surface-shift test, language-shift test. | [phase_02_dataset_preparation.md](phase_results/phase_02_dataset_preparation.md) |
| 3 | Pilot activation extraction | Complete | Extracted pilot activations; context-length ablation showed `ctx512` was the practical default. | [phase_03_activation_extraction.md](phase_results/phase_03_activation_extraction.md) |
| 4 | Metrics and baselines | Complete | Implemented FVE/MSE/RMSE/cosine and mean/zero/shuffled baselines. | [phase_04_metrics_and_baselines.md](phase_results/phase_04_metrics_and_baselines.md) |
| 5 | Initial AR baseline | Complete | DistilBERT AR ran end-to-end but stayed below mean baseline on the tiny pilot. | [phase_05_ar_baseline.md](phase_results/phase_05_ar_baseline.md) |
| 5b | AR diagnostics | Complete | Standardization improved AR; first positive pilot AR FVE appeared, but debug AR remained limited. | [phase_07b_ar_diagnostics.md](phase_results/phase_07b_ar_diagnostics.md) |
| 6a | Scaled activation extraction | Complete | Extracted Qwen 1.5B layer-19 ctx512 activations for train and validation. | [phase_06_scaled_activation_extraction.md](phase_results/phase_06_scaled_activation_extraction.md) |
| 6b | Scaled baselines | Complete | Established train/validation baseline metrics using train-reference mean. | [phase_06b_scaled_baselines.md](phase_results/phase_06b_scaled_baselines.md) |
| 6c | Scaled AR baselines | Complete | DistilBERT AR reached FVE `0.095719` on reference descriptions and `0.238927` on code text. | [refdesc](phase_results/phase_06c_scaled_ar_refdesc.md), [code](phase_results/phase_06c_scaled_ar_code.md) |
| 7 | Debug AV baseline | Complete | DistilGPT2 AV trained on reference descriptions; best validation loss `3.098751`. | [phase_07_av_baseline.md](phase_results/phase_07_av_baseline.md) |
| 8 | Debug full NLA loop | Complete | Full Distil debug loop executed, but failed mean baseline: validation FVE `-0.353821`. | [phase_08_full_nla_loop.md](phase_results/phase_08_full_nla_loop.md) |
| 9a | Test activation extraction | Complete | Extracted activations for `test_indomain`, `test_surface_shift`, and `test_language_shift`. | [phase_09a_test_activation_extraction.md](phase_results/phase_09a_test_activation_extraction.md) |
| 9b | Debug controlled tests | Complete | Debug loop failed all controlled tests; useful as a negative baseline. | [phase_09b_test_full_loop.md](phase_results/phase_09b_test_full_loop.md) |
| 10a | Qwen AR/AV smoke | Complete | Qwen 0.5B LoRA AR and AV smoke tests succeeded. | [phase_10a_qwen_aligned_smoke.md](phase_results/phase_10a_qwen_aligned_smoke.md) |
| 10c-smoke | Qwen NLA smoke/adaptation | Complete | Qwen 0.5B full loop and generated-text AR adaptation worked, but tiny smoke did not improve quality. | [phase_10c_qwen_nla_adaptation_smoke.md](phase_results/phase_10c_qwen_nla_adaptation_smoke.md) |
| 10c-medium | Medium Qwen aligned run | Complete | Qwen 0.5B after adaptation reached validation FVE `0.494062`, MSE `0.131743`. | [phase_10c_qwen_medium_aligned_run.md](phase_results/phase_10c_qwen_medium_aligned_run.md) |
| 10c-tests | Medium Qwen test evaluation | Complete | Beat mean baseline on in-domain and surface-shift; language-shift remained weak. | [phase_10c_qwen_medium_test_evaluation.md](phase_results/phase_10c_qwen_medium_test_evaluation.md) |
| 10d | Final Qwen 1.5B aligned run | Complete | Full train/validation aligned run: FVE `0.361623`, MSE `0.149737`, beating mean baseline. | [phase_10d_qwen15b_joint_run.md](phase_results/phase_10d_qwen15b_joint_run.md) |
| 11-train | Reward-driven AV optimization | Complete | Reconstruction-reward AV training improved validation to FVE `0.457392`, MSE `0.127274`. | [phase_11_qwen15b_av_reward_rl.md](phase_results/phase_11_qwen15b_av_reward_rl.md) |
| 11-tests | Final controlled tests | Complete | Final RL system beat mean baseline on in-domain and surface-shift; failed language-shift. | [phase_11_qwen15b_av_reward_rl_test_evaluation.md](phase_results/phase_11_qwen15b_av_reward_rl_test_evaluation.md) |
| 12 | Final report | Complete | README and supporting documentation prepared for submission. | [README.md](../README.md) |

## Dataset and artifact milestones

### Processed datasets

| Split | Rows | Purpose |
|---|---:|---|
| `pilot_100.jsonl` | 100 | Fast smoke tests and context ablations. |
| `train.jsonl` | 5000 | Main training split. |
| `validation.jsonl` | 500 | Model selection and validation metrics. |
| `test_indomain.jsonl` | 500 | Held-out Python in-domain evaluation. |
| `test_surface_shift.jsonl` | 500 | Identifier-renaming / surface-level robustness. |
| `test_language_shift.jsonl` | 361 | Cross-language generalization stress test. |

### Main activation artifacts

| Artifact | Shape | Notes |
|---|---:|---|
| `train_qwen25_coder_15b_l19_ctx512` | `(5000, 1536)` | Main train activations. |
| `validation_qwen25_coder_15b_l19_ctx512` | `(500, 1536)` | Main validation activations. |
| `test_indomain_qwen25_coder_15b_l19_ctx512` | `(500, 1536)` | In-domain test activations. |
| `test_surface_shift_qwen25_coder_15b_l19_ctx512` | `(500, 1536)` | Surface-shift test activations. |
| `test_language_shift_qwen25_coder_15b_l19_ctx512` | `(361, 1536)` | Language-shift test activations. |

Large artifacts are not committed; these names document the local output directories used in the experiments.

## Key result progression

### Validation progression

| System | Validation FVE | Validation MSE | Interpretation |
|---|---:|---:|---|
| Debug Distil full loop | -0.353821 | 0.317551 | End-to-end path works but weak. |
| Medium Qwen 0.5B after adaptation | 0.494062 | 0.131743 | Strong proof of concept. |
| Final Qwen 1.5B aligned joint | 0.361623 | 0.149737 | Full data, full validation, beats mean baseline. |
| Final Qwen 1.5B reward-RL AV | 0.457392 | 0.127274 | Best final validation result. |

### Final controlled test results

| Split | FVE | MSE | Mean MSE | Outcome |
|---|---:|---:|---:|---|
| `test_indomain` | 0.400884 | 0.079164 | 0.132134 | Success. |
| `test_surface_shift` | 0.480390 | 0.102624 | 0.197502 | Success. |
| `test_language_shift` | -4.647290 | 0.107910 | 0.019108 | Limitation. |

## Main decisions and conclusions

1. **Context length:** `max_length=512` was selected because it greatly reduced truncation while staying practical on the available GPU.
2. **Model family:** Qwen-based AV/AR was necessary; DistilBERT/DistilGPT2 was useful only as a debug baseline.
3. **Target transform:** activation standardization was important for stable AR training.
4. **Distribution mismatch:** supervised AV plus supervised AR was insufficient; AR had to see AV-generated text.
5. **Reward stage:** reward-driven AV optimization was added to move beyond supervised imitation and toward the original NLA objective.
6. **Generalization:** in-domain and surface-shift succeeded; language-shift remains the primary limitation.

## Final status

The project is complete for submission. The final report is [README.md](../README.md), with supporting details in [docs/phase_results/](phase_results/), [docs/research_log.md](research_log.md), and [experiments/](../experiments/).
