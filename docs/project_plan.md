# Project Plan

This document is the active roadmap for the project. These phase numbers are the phase numbers used from now on.

## Current Phase

**Current phase:** Phase 5b — Scale activation extraction for AR training.

**Immediate next step:** extract train and validation activations for the main model with `max_length=512`.

## Phase 1 — Define research question and scope

**Goal:** Define the project as a compact AI4Code adaptation of Natural Language Autoencoders.

**Research question:**

> Can a simplified NLA recover meaningful information from residual-stream activations of a small code language model, and do the resulting explanations remain stable under surface-level and programming-language shifts?

**Key decisions:**

- Target model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Smoke-test model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Task: function-level code understanding
- Activation type: residual-stream hidden state
- Token position: final non-padding token
- Main-model layer: `19`
- Smoke-model layer: `16`
- Metric: Fraction of Variance Explained, FVE

**Status:** Complete.

## Phase 2 — Build environment and verify model access

**Goal:** Validate CUDA, model loading, hidden-state extraction, and `inputs_embeds` compatibility.

**Main files:**

- `docs/setup_and_model_download.md`
- `scripts/feasibility_probe.py`
- `src/nla_code_interp/activations.py`

**Results:**

| Model | Layer | Hidden size | Result |
|---|---:|---:|---|
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 16 | 896 | Success |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 19 | 1536 | Success |

**Report:** `docs/phase_results/phase_01_feasibility_probe.md`

**Status:** Complete.

## Phase 3 — Prepare datasets

**Goal:** Convert raw datasets into project-standard JSONL files.

**Main files:**

- `scripts/prepare_dataset.py`
- `src/nla_code_interp/data.py`

**Outputs:**

| File | Rows |
|---|---:|
| `pilot_100.jsonl` | 100 |
| `train.jsonl` | 5000 |
| `validation.jsonl` | 500 |
| `test_indomain.jsonl` | 500 |
| `test_surface_shift.jsonl` | 500 |
| `test_language_shift.jsonl` | 361 |

**Report:** `docs/phase_results/phase_02_dataset_preparation.md`

**Status:** Complete.

## Phase 4 — Extract and validate pilot activations

**Goal:** Extract pilot activations and select a context length.

**Main files:**

- `scripts/extract_activations.py`
- `src/nla_code_interp/activations.py`

**Pilot artifacts:**

| Artifact | Model | Layer | Max length | Examples | Shape | Truncated |
|---|---|---:|---:|---:|---|---:|
| `pilot_100_qwen25_coder_05b_l16` | 0.5B | 16 | 128 | 100 | `(100, 896)` | 66 |
| `pilot_100_qwen25_coder_15b_l19` | 1.5B | 19 | 128 | 100 | `(100, 1536)` | 66 |
| `pilot_100_qwen25_coder_15b_l19_ctx256` | 1.5B | 19 | 256 | 100 | `(100, 1536)` | 30 |
| `pilot_100_qwen25_coder_15b_l19_ctx512` | 1.5B | 19 | 512 | 100 | `(100, 1536)` | 1 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | 1.5B | 19 | 1024 | 100 | `(100, 1536)` | 0 |

**Decision:** Use `max_length=512` for main-model extraction.

**Report:** `docs/phase_results/phase_03_activation_extraction.md`

**Status:** Complete.

## Phase 5 — Metrics, baselines, and AR pilot

**Goal:** Validate reconstruction metrics and build the first AR baseline.

### Phase 5a — Metrics and baselines

**Main files:**

- `scripts/run_evaluation.py`
- `src/nla_code_interp/metrics.py`

**Results:**

| Artifact | Mean FVE | Zero FVE | Shuffled FVE |
|---|---:|---:|---:|
| `pilot_100_qwen25_coder_15b_l19_ctx512` | 0.000000 | -51.401844 | -1.014737 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | 0.000000 | -93.230362 | -1.001524 |
| `pilot_100_qwen25_coder_05b_l16` | 0.000000 | -0.668432 | -0.920304 |

**Report:** `docs/phase_results/phase_04_metrics_and_baselines.md`

**Status:** Complete.

### Phase 5b — AR pilot and diagnostics

**Main files:**

- `scripts/train_ar.py`
- `src/nla_code_interp/models.py`

**Results:**

| Setup | Text field | Frozen | Target transform | Best epoch | Validation FVE | Validation MSE | Beats train-mean baseline? |
|---|---|---|---|---:|---:|---:|---|
| refdesc raw | `reference_description` | yes | raw | 20 | -5.110613 | 0.083229 | no |
| refdesc center | `reference_description` | yes | center | 8 | -0.272226 | 0.017328 | no |
| refdesc standardize | `reference_description` | yes | standardize | 13 | 0.056828 | 0.012846 | yes |
| code center | `code` | yes | center | 8 | -0.032409 | 0.014062 | yes |
| refdesc center unfrozen | `reference_description` | no | center | 10 | -0.099820 | 0.014980 | no |

**Decision:** Use target standardization for AR training. Best pilot setting: `reference_description + standardize + frozen DistilBERT`.

**Reports:**

- `docs/phase_results/phase_05_ar_baseline.md`
- `docs/phase_results/phase_07b_ar_diagnostics.md`

**Status:** In progress. Pilot diagnostics are complete; scaled train and validation activations are needed next.

## Phase 6 — Scale activations and train AR on larger data

**Goal:** Move from pilot AR to train/validation AR.

**Steps:**

1. Extract train activations with the main model:
   - input: `data/processed/train.jsonl`
   - model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
   - layer: `19`
   - max length: `512`
2. Extract validation activations with the same setup.
3. Run baseline evaluation on both artifacts.
4. Train AR using:
   - text model: `distilbert-base-uncased`
   - text field: `reference_description`
   - target transform: `standardize`
   - frozen text encoder
5. Run one comparison using `text_field=code`.

**Status:** Next phase.

## Phase 7 — Implement AV

**Goal:** Train the activation-to-text model.

**Dependency:** Start after AR is trained and validated on larger train/validation artifacts.

**Status:** Not started.

## Phase 8 — Connect the full NLA loop

**Goal:** Run:

```text
activation -> AV -> explanation -> AR -> reconstructed activation -> FVE
```

**Status:** Not started.

## Phase 9 — Controlled evaluations

**Goal:** Evaluate in-domain, surface-shift, and language-shift behavior.

**Planned comparisons:**

- In-domain Python
- Identifier-renamed Python
- Formatting/comment changes
- Python, C++, and Java language shift
- Context length
- Text source
- Model size, if time permits

**Status:** Not started.

## Phase 10 — Final report

**Goal:** Produce the final README/report, tables, figures, limitations, and reproducibility commands.

**Status:** Not started.
