# Natural Language Autoencoders for Code-Semantic Activations

This repository contains a compact, reproducible reimplementation plan for the KTH PhD recruitment task on **Natural Language Autoencoders (NLA)** applied to small open-source code language models.

The project investigates whether a simplified NLA can translate internal activations of a code language model into natural-language explanations, and whether those explanations preserve information about code semantics under surface-level and programming-language shifts.

## Research question

Can a simplified natural language autoencoder recover meaningful information from the residual-stream activations of a small code language model, and do the resulting explanations reflect code semantics rather than only superficial syntax?

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
├── docs/                    # Project plan, task definition, setup, and research notes
├── experiments/             # Experiment registry CSV and lightweight metadata
├── notebooks/               # Optional exploratory notebooks
├── prompts/                 # Codex/agent prompts used during implementation
├── scripts/                 # CLI entry points for each pipeline stage
├── src/nla_code_interp/     # Python package skeleton
├── tests/                   # Basic smoke tests
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

This repository currently contains the project scaffold. The next implementation step is to validate the CUDA environment, load the smoke-test model, and verify that hidden states can be extracted from the selected layer.

## Reproducibility principle

Large artifacts such as model weights, raw datasets, extracted activations, checkpoints, and generated reports should stay outside Git and be referenced through documented local paths or release artifacts.
