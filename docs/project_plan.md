# Project Plan

This document is the living roadmap for the project. It tracks the official project phases, implementation decisions, completed work, deviations from the initial plan, and next actions.

## Current Status Summary

The project has completed the setup, dataset preparation, pilot activation extraction, context-length ablation, baseline evaluation, and first AR diagnostics. The current bottleneck is no longer infrastructure; it is scaling AR training beyond the 100-example pilot artifact.

Recommended next step:

> Extract train/validation activations for the main model with `max_length=512`, then retrain AR on the larger activation set before implementing AV.

## Naming Note

Some early chat/execution messages used a separate execution-step numbering, for example “Phase 1 feasibility probe” and “Phase 5 AR baseline.” The official phase numbering in this file is the source of truth going forward.

Approximate mapping:

| Chat / execution label | Official project-plan phase |
|---|---|
| Feasibility probe | Phase 3 |
| Dataset preparation | Phase 4 |
| Activation extraction | Phase 5 |
| Metrics and baselines | Phase 6 |
| AR baseline / AR diagnostics | Phase 7 |

## Phase 1 — Understand the task and define the research question

**Goal:** Translate the recruitment task into a focused, testable research problem.

**Current research question:**

> Can a simplified natural language autoencoder recover meaningful information from residual-stream activations of a small code language model, and do the resulting explanations remain stable under surface-level and programming-language shifts?

**Status:** Complete.

**Notes:**

- The project is positioned as a compact AI4Code adaptation of NLA.
- The target is not a full reproduction of the official large-scale NLA stack.
- The implementation should remain transparent, reproducible, and suitable for a single-GPU setting.

## Phase 2 — Lock the experimental scope

**Goal:** Select the target model, activation location, task domain, datasets, evaluation criteria, and compute budget.

**Decisions:**

- Target model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Smoke-test model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Primary task: function-level code understanding
- Training domain: Python functions
- Controlled evaluation: in-domain Python, identifier renaming, and multilingual code variants
- Activation type: residual-stream hidden state
- Token position: final non-padding token of the prompt
- Target layer for 0.5B model: layer index `16`
- Target layer for 1.5B model: layer index `19`
- Primary metric: Fraction of Variance Explained (FVE)
- Main hardware used so far: NVIDIA RTX 3090-class GPU

**Status:** Complete, with decisions validated by later feasibility and extraction runs.

## Phase 3 — Build the reproducible environment

**Goal:** Validate CUDA, dependency versions, model loading, hidden-state extraction, and local storage conventions.

**Deliverables:**

- Environment setup instructions: `docs/setup_and_model_download.md`
- Source usage policy: `docs/source_usage.md`
- Official-to-local mapping: `docs/reference_mapping.md`
- CUDA/model feasibility probe: `scripts/feasibility_probe.py`

**Completed results:**

- `Qwen/Qwen2.5-Coder-0.5B-Instruct` loaded successfully.
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` loaded successfully.
- Hidden states were returned with expected shape.
- Final non-padding token activation selection works.
- `inputs_embeds` compatibility check passed, which is important for later AV design.

**Report:** `docs/phase_results/phase_01_feasibility_probe.md`

**Status:** Complete.

## Phase 4 — Prepare the dataset

**Goal:** Create reproducible train, validation, and test splits for code-semantic activation extraction.

**Deliverables:**

- Dataset preparation script: `scripts/prepare_dataset.py`
- Dataset helpers: `src/nla_code_interp/data.py`
- Processed local JSONL outputs under `data/processed/` — not committed to Git
- Dataset manifest under `data/processed/dataset_manifest.json` — not committed to Git

**Completed outputs:**

| File | Rows |
|---|---:|
| `pilot_100.jsonl` | 100 |
| `train.jsonl` | 5000 |
| `validation.jsonl` | 500 |
| `test_indomain.jsonl` | 500 |
| `test_surface_shift.jsonl` | 500 |
| `test_language_shift.jsonl` | 361 |

**Notes:**

- HumanEval-X has fewer filtered examples than the requested `test_size=500` for some languages. This is expected and documented.
- Surface-shift examples currently use conservative identifier renaming.

**Report:** `docs/phase_results/phase_02_dataset_preparation.md`

**Status:** Complete.

## Phase 5 — Extract and verify activations

**Goal:** Extract target-model activations and verify that they are correct, stable, and informative.

**Deliverables:**

- Activation extraction script: `scripts/extract_activations.py`
- Activation utilities: `src/nla_code_interp/activations.py`
- Local activation artifacts under `outputs/activations/` — not committed to Git

**Completed pilot artifacts:**

| Artifact | Model | Layer | Max length | Examples | Activation shape | Truncated prompts |
|---|---|---:|---:|---:|---|---:|
| `pilot_100_qwen25_coder_05b_l16` | 0.5B | 16 | 128 | 100 | `(100, 896)` | 66 |
| `pilot_100_qwen25_coder_15b_l19` | 1.5B | 19 | 128 | 100 | `(100, 1536)` | 66 |
| `pilot_100_qwen25_coder_15b_l19_ctx256` | 1.5B | 19 | 256 | 100 | `(100, 1536)` | 30 |
| `pilot_100_qwen25_coder_15b_l19_ctx512` | 1.5B | 19 | 512 | 100 | `(100, 1536)` | 1 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | 1.5B | 19 | 1024 | 100 | `(100, 1536)` | 0 |

**Context-length decision:**

- `max_length=512` is the recommended default for the main model because it nearly eliminates truncation while remaining efficient.
- `max_length=1024` is kept as a no-truncation comparison point.

**Report:** `docs/phase_results/phase_03_activation_extraction.md`

**Status:** Pilot complete. Full train/validation extraction is the next recommended step.

## Phase 6 — Implement baselines and metrics

**Goal:** Establish reference points before training the full NLA.

**Deliverables:**

- Baseline evaluation script: `scripts/run_evaluation.py`
- Metrics: `src/nla_code_interp/metrics.py`
- Local reports under `outputs/reports/baselines/` — not committed to Git

**Implemented baselines:**

- Mean activation prediction
- Zero activation prediction
- Shuffled activation prediction

**Completed pilot results:**

| Artifact | Mean FVE | Zero FVE | Shuffled FVE |
|---|---:|---:|---:|
| `pilot_100_qwen25_coder_15b_l19_ctx512` | 0.000000 | -51.401844 | -1.014737 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | 0.000000 | -93.230362 | -1.001524 |
| `pilot_100_qwen25_coder_05b_l16` | 0.000000 | -0.668432 | -0.920304 |

**Interpretation:**

- Mean baseline FVE is 0 by construction when computed from the same target tensor.
- Negative zero and shuffled baselines indicate the FVE implementation behaves as expected.

**Report:** `docs/phase_results/phase_04_metrics_and_baselines.md`

**Status:** Complete for pilot artifacts.

## Phase 7 — Implement and train the Activation Reconstructor (AR)

**Goal:** Train a text-to-activation model and validate that the reconstruction pipeline works independently.

**Implemented so far:**

- `TextActivationReconstructor` in `src/nla_code_interp/models.py`
- AR training script: `scripts/train_ar.py`
- Frozen DistilBERT text encoder baseline
- Mean pooling over text hidden states
- Projection head to activation dimension
- Deterministic pilot train/validation split
- Target transformation diagnostics: raw, center, and standardize

### Phase 7a — Pilot AR baseline

**Artifact:** `pilot_100_qwen25_coder_15b_l19_ctx512`

**First baseline setup:**

- Text model: `distilbert-base-uncased`
- Text field: `reference_description`
- Text model frozen: yes
- Target transform: raw
- Train/validation split: 80 / 20

**Result:**

- Best validation FVE: `-5.110613`
- Best validation RMSE: `0.288495`
- Best validation cosine mean: `0.979651`

**Interpretation:**

- Training runs end-to-end.
- High cosine but negative FVE suggested a scale/centering issue.
- The 100-example pilot is not enough for a final judgment of AR quality.

**Report:** `docs/phase_results/phase_05_ar_baseline.md`

### Phase 7b — AR diagnostics on pilot

**Completed diagnostic runs:**

| Setup | Text field | Frozen | Target transform | Best epoch | Validation FVE | Validation MSE | Validation cosine | Beats train-mean baseline? |
|---|---|---|---|---:|---:|---:|---:|---|
| refdesc center | `reference_description` | yes | center | 8 | -0.272226 | 0.017328 | 0.994898 | no |
| refdesc standardize | `reference_description` | yes | standardize | 13 | 0.056828 | 0.012846 | 0.996221 | yes |
| code center | `code` | yes | center | 8 | -0.032409 | 0.014062 | 0.995905 | yes |
| refdesc center unfrozen | `reference_description` | no | center | 10 | -0.099820 | 0.014980 | 0.995556 | no |

**Interpretation:**

- Standardizing activation targets is the strongest pilot result so far.
- The first positive AR validation FVE is `0.056828`.
- Code text with centered targets also beats the validation train-mean baseline, even though its FVE remains slightly negative.
- Unfreezing DistilBERT for this small pilot did not help.
- The next AR experiment should use more than 100 examples before drawing strong conclusions.

**Report:** `docs/phase_results/phase_07b_ar_diagnostics.md`

**Status:** Pilot diagnostics complete. Needs scaled train/validation activations.

## Phase 8 — Implement and train the Activation Verbalizer (AV)

**Goal:** Train an activation-to-text model that produces compact natural-language explanations.

**Status:** Not started.

**Dependency:** Do not start this phase until AR is trained on a larger train/validation activation set and has a reasonable validation baseline.

## Phase 9 — Connect the NLA loop

**Goal:** Run the complete vector-to-text-to-vector pipeline and measure reconstruction quality.

**Status:** Not started.

**Dependency:** Requires a usable AR and a first AV implementation.

## Phase 10 — Run controlled experiments

**Planned comparisons:**

- In-domain Python code
- Identifier-renamed Python code
- Formatting and comment changes
- Same-semantics code across Python, C++, and Java
- Layer and explanation-length ablations
- Context-length ablations
- Text-source ablations for AR

**Status:** Partially started through pilot context-length and AR text-source diagnostics. Full controlled evaluation is not started.

## Phase 11 — Analyze results

**Goal:** Combine quantitative reconstruction results with qualitative explanation analysis.

**Questions:**

- Does high FVE correspond to meaningful explanations?
- Does the NLA capture code semantics or superficial syntax?
- Which examples fail, and why?
- How does performance change under distribution shift?

**Status:** Early numerical analysis only. Qualitative AV analysis is not started.

## Phase 12 — Prepare the final report

**Goal:** Produce a concise, honest, reproducible README and supporting figures.

**Final deliverables:**

- Reproducible code
- Experiment registry
- Quantitative tables and plots
- Qualitative examples and failure modes
- Final README within the recruitment-task word limit

**Status:** Not started.

## Immediate Next Actions

1. Extract main-model train activations with `max_length=512`.
2. Extract main-model validation activations with `max_length=512`.
3. Run baseline evaluation on train and validation artifacts.
4. Train AR on larger train/validation artifacts using `target_transform=standardize`.
5. Compare `reference_description`, `code`, and `prompt` text sources.
6. Only then start AV design and activation-to-text training.
