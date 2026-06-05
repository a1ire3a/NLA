# Natural Language Autoencoders for Code-Semantic Activations

This repository contains a compact, reproducible implementation for the KTH PhD recruitment task on **Natural Language Autoencoders (NLA)** applied to small open-source code language models.

The project investigates whether a simplified NLA can translate internal activations of a code language model into natural-language explanations, and whether those explanations preserve information about code semantics under surface-level and programming-language shifts.

## Research question

Can a simplified natural language autoencoder recover meaningful information from the residual-stream activations of a small code language model, and do the resulting explanations reflect code semantics rather than only superficial syntax?

## Implementation stance

This repository is **not** a fork of the official NLA repository. It is a clean, small-scale implementation inspired by the official NLA work and adapted to a single-GPU AI4Code setting.

The official implementation is treated as a reference for terminology, architecture, and evaluation design:

- Official NLA repository: https://github.com/kitft/natural_language_autoencoders
- Source usage policy: `docs/source_usage.md`
- Local-to-official design mapping: `docs/reference_mapping.md`

If code is copied or closely adapted from the official repository, the relevant file must include explicit attribution and license notes.

## Initial scope

- **Target model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Smoke-test model:** `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- **Task domain:** function-level code understanding
- **Primary training data:** CodeSearchNet / CodeXGLUE-style Python function examples
- **Controlled test data:** HumanEval-X / multilingual function-level code examples
- **Activation target:** residual-stream hidden states from a selected transformer layer
- **Primary metric:** Fraction of Variance Explained (FVE)

## Repository structure

```text
.
├── configs/                 # YAML experiment configurations
├── data/                    # Dataset notes and local data layout; raw data is not committed
├── docs/                    # Project plan, setup notes, phase reports, and research log
├── experiments/             # Experiment registry CSV and lightweight metadata
├── notebooks/               # Optional exploratory notebooks
├── prompts/                 # Codex/agent prompts used during implementation
├── scripts/                 # CLI entry points for each pipeline stage
├── src/nla_code_interp/     # Python package
├── tests/                   # Smoke and unit tests
├── requirements.txt
└── README.md
```

## Pipeline overview

```text
code prompt
  -> target code LLM
  -> layer activation
  -> activation verbalizer (AV): vector -> text
  -> activation reconstructor (AR): text -> vector
  -> reconstruction score and qualitative analysis
```

## Current status

The debug baseline pipeline is complete end-to-end:

```text
activation extraction -> metrics -> AR -> AV -> full loop -> controlled test evaluation
```

The DistilBERT/DistilGPT2 debug baseline successfully validated the implementation, but it did not produce final-quality reconstruction. The current project phase is the Qwen-based aligned NLA implementation:

- Qwen AR LoRA smoke test: complete
- Qwen AV LoRA smoke test: complete
- Qwen AV generation smoke test: complete
- Next step: reconstruction-aware / aligned Qwen NLA training with the 0.5B model before final 1.5B runs

Primary planning and reporting files:

- Active roadmap: `docs/project_plan.md`
- Central narrative log: `docs/research_log.md`
- Experiment registry: `experiments/experiment_log.csv`
- Phase-specific reports: `docs/phase_results/`

## Reproducibility principle

Large artifacts such as model weights, raw datasets, extracted activations, checkpoints, and generated reports should stay outside Git and be referenced through documented local paths or release artifacts.
