# Project Plan

This document summarizes the project roadmap, main experiment phases, and final technical status for the NLA-for-code-activations project.

## Current Status

**Current phase:** Final documentation and public release.

**Documentation state:** complete. The repository contains the core code, README, phase reports, experiment registries, and reproduction instructions.

Main entry points:

- [README.md](../README.md) — short project report.
- [docs/research_log.md](research_log.md) — narrative record of decisions, results, and issues.
- [docs/phase_results/](phase_results/) — detailed phase-by-phase reports.
- [experiments/](../experiments/) — lightweight CSV experiment registries.
- [docs/manual_installation.md](manual_installation.md) — setup and reproduction guide.

## Phase Timeline

| Phase | Title | Status | Key Output |
|---|---|---|---|
| 1 | Feasibility probe | Complete | Verified Qwen model loading, hidden-state extraction, and `inputs_embeds` support. |
| 2 | Dataset preparation | Complete | Built 6961 processed examples across train, validation, and controlled test splits. |
| 3 | Activation extraction | Complete | Extracted Qwen2.5-Coder layer-19 activations and selected `ctx512` as the practical default. |
| 4 | Metrics and baselines | Complete | Implemented FVE, MSE, RMSE, cosine, and trivial reconstruction baselines. |
| 5 | Debug AR/AV baselines | Complete | Built DistilBERT/DistilGPT2 baselines to validate the end-to-end loop. |
| 6 | Scaled activation experiments | Complete | Extracted train/validation activations and evaluated scaled AR baselines. |
| 7 | Full debug NLA loop | Complete | Verified the full activation-to-text-to-activation loop, but it stayed below mean baseline. |
| 8 | Controlled evaluations | Complete | Evaluated in-domain, surface-shift, and language-shift splits. |
| 9 | Qwen-based AR/AV | Complete | Replaced weak debug components with Qwen-based LoRA modules. |
| 10 | Aligned Qwen runs | Complete | Improved reconstruction after generated-text AR adaptation. |
| 11 | Reward-driven AV optimization | Complete | Improved validation FVE from `0.361623` to `0.457392`. |
| 12 | Final documentation | Complete | Prepared README, phase reports, and reproduction notes for public presentation. |

## Dataset and Artifact Milestones

| Split | Rows | Purpose |
|---|---:|---|
| `pilot_100.jsonl` | 100 | Fast smoke tests and context ablations. |
| `train.jsonl` | 5000 | Main training split. |
| `validation.jsonl` | 500 | Model selection and validation metrics. |
| `test_indomain.jsonl` | 500 | Held-out Python in-domain evaluation. |
| `test_surface_shift.jsonl` | 500 | Identifier-renaming robustness evaluation. |
| `test_language_shift.jsonl` | 361 | Cross-language generalization stress test. |

Large artifacts such as raw datasets, extracted activations, and checkpoints are intentionally not committed. Their local paths and generation commands are documented in the reproduction guide.

## Key Result Progression

| System | Validation FVE | Validation MSE | Interpretation |
|---|---:|---:|---|
| Debug Distil full loop | -0.353821 | 0.317551 | End-to-end path worked but was weak. |
| Medium Qwen 0.5B after adaptation | 0.494062 | 0.131743 | Strong proof of concept. |
| Final Qwen 1.5B aligned joint | 0.361623 | 0.149737 | Full-data run beat the mean baseline. |
| Final Qwen 1.5B reward-RL AV | 0.457392 | 0.127274 | Best final validation result. |

## Final Controlled Test Results

| Split | FVE | MSE | Mean MSE | Outcome |
|---|---:|---:|---:|---|
| `test_indomain` | 0.400884 | 0.079164 | 0.132134 | Success. |
| `test_surface_shift` | 0.480390 | 0.102624 | 0.197502 | Success. |
| `test_language_shift` | -4.647290 | 0.107910 | 0.019108 | Limitation. |

## Main Decisions and Conclusions

1. `max_length=512` was selected because it reduced truncation while staying practical on the available GPU.
2. Qwen-based AR/AV components were necessary; DistilBERT/DistilGPT2 were useful only as debug baselines.
3. Activation standardization was important for stable AR training.
4. AR needed exposure to AV-generated text to reduce distribution mismatch.
5. Reward-driven AV optimization produced the strongest final validation result.
6. In-domain and surface-shift reconstruction succeeded, while language-shift remained the main limitation.

## Final Status

The project documentation is complete. The short project report is [README.md](../README.md), with supporting details in [docs/phase_results/](phase_results/), [docs/research_log.md](research_log.md), and [experiments/](../experiments/).
