# Project Plan

This document is the living roadmap for the project. Each phase should be updated with decisions, implementation notes, results, and unresolved questions.

## Phase 1 — Understand the task and define the research question

**Goal:** Translate the recruitment task into a focused, testable research problem.

**Current research question:**

> Can a simplified natural language autoencoder recover meaningful information from residual-stream activations of a small code language model, and do the resulting explanations remain stable under surface-level and programming-language shifts?

**Status:** Complete.

## Phase 2 — Lock the experimental scope

**Goal:** Select the target model, activation location, task domain, datasets, evaluation criteria, and compute budget.

**Initial decisions:**

- Target model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Smoke-test model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Primary task: function-level code understanding
- Training domain: Python functions
- Controlled evaluation: in-domain Python, identifier renaming, and multilingual code variants
- Activation type: residual-stream hidden state
- Token position: final non-padding token of the prompt
- Initial target layer: approximately two-thirds through the model
- Primary metric: Fraction of Variance Explained (FVE)

**To complete:**

- Confirm exact model revision and license
- Confirm exact layer index after feasibility probing
- Confirm data source and filtering rules
- Confirm expected training sample counts

## Phase 3 — Build the reproducible environment

**Goal:** Validate CUDA, dependency versions, model loading, hidden-state extraction, and local storage conventions.

**Deliverables:**

- Environment setup instructions
- CUDA feasibility probe
- Model download instructions
- Local directory conventions

## Phase 4 — Prepare the dataset

**Goal:** Create reproducible train, validation, and test splits for code-semantic activation extraction.

**Deliverables:**

- Dataset preparation script
- Filtering and deduplication rules
- Prompt construction logic
- Dataset cards and statistics

## Phase 5 — Extract and verify activations

**Goal:** Extract target-model activations and verify that they are correct, stable, and informative.

**Verification checklist:**

- Correct layer and token position
- Stable tensor shape
- No padding-token contamination
- Deterministic extraction under a fixed seed
- Non-trivial variation across examples
- Reasonable activation norms and distribution

## Phase 6 — Implement baselines and metrics

**Goal:** Establish reference points before training the full NLA.

**Baselines:**

- Mean activation prediction
- Random activation prediction
- Shuffled explanation reconstruction
- Direct numerical reconstruction baseline

**Metric:**

- Fraction of Variance Explained (FVE)

## Phase 7 — Implement and train the Activation Reconstructor (AR)

**Goal:** Train a text-to-activation model and validate that the reconstruction pipeline works independently.

## Phase 8 — Implement and train the Activation Verbalizer (AV)

**Goal:** Train an activation-to-text model that produces compact natural-language explanations.

## Phase 9 — Connect the NLA loop

**Goal:** Run the complete vector-to-text-to-vector pipeline and measure reconstruction quality.

## Phase 10 — Run controlled experiments

**Planned comparisons:**

- In-domain Python code
- Identifier-renamed Python code
- Formatting and comment changes
- Same-semantics code across Python, C++, and Java
- Layer and explanation-length ablations

## Phase 11 — Analyze results

**Goal:** Combine quantitative reconstruction results with qualitative explanation analysis.

**Questions:**

- Does high FVE correspond to meaningful explanations?
- Does the NLA capture code semantics or superficial syntax?
- Which examples fail, and why?
- How does performance change under distribution shift?

## Phase 12 — Prepare the final report

**Goal:** Produce a concise, honest, reproducible README and supporting figures.

**Final deliverables:**

- Reproducible code
- Experiment registry
- Quantitative tables and plots
- Qualitative examples and failure modes
- Final README within the recruitment-task word limit
