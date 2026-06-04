# Experiment Plan

This document will be expanded throughout the project. It records the scientific decisions, implementation steps, and evaluation plan for the NLA-for-code-semantics study.

## 1. Research Goal

The project investigates whether a simplified Natural Language Autoencoder (NLA) can interpret hidden-state activations from a small open-source Code LLM.

Working question:

> Can natural-language explanations generated from a selected activation preserve enough information to reconstruct that activation, and do those explanations reflect code semantics rather than only surface syntax?

## 2. Initial Hypothesis

A code-specialized LLM should encode functional information about a code snippet in intermediate residual-stream activations. If an AV model can verbalize part of that information, an AR model should be able to reconstruct a non-trivial fraction of the original activation from the generated explanation.

## 3. Initial Model Choices

- Smoke-test model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Main model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`

The smoke-test model is used to debug the full pipeline cheaply. The main model is used for the final experiments if compute permits.

## 4. Initial Data Choices

- Training-style data: Python function-level examples from CodeSearchNet / CodeXGLUE-style sources.
- Controlled evaluation data: HumanEval-X or similar multilingual function-level code examples.

The first implementation should use a small pilot subset before downloading or processing large datasets.

## 5. Activation Extraction Protocol

Initial protocol:

- Use Hugging Face Transformers, not Ollama or GGUF models.
- Enable `output_hidden_states=True`.
- Extract residual-stream hidden states from a selected transformer layer.
- Use the final non-padding token of a standardized prompt.
- Store activations outside Git under `outputs/activations/` or a configured local artifact directory.

Initial layer choice:

- For a 24-layer model: around layer 16.
- For a 28-layer model: around layer 19.

This corresponds roughly to two-thirds through the model depth.

## 6. Planned Baselines

- Mean activation baseline.
- Random activation baseline.
- Shuffled explanation baseline.
- Direct text-to-activation AR baseline using the original code prompt.

## 7. Primary Metric

Fraction of Variance Explained (FVE):

```text
FVE = 1 - SSE(reconstructed, original) / SSE(mean_baseline, original)
```

FVE should be computed on held-out test activations.

## 8. Qualitative Analysis

For selected examples, record:

- Original code snippet.
- AV explanation.
- Reference description, if available.
- FVE score.
- Whether the explanation seems semantic, syntactic, generic, or hallucinated.

## 9. Planned Ablations

- Model size: 0.5B vs 1.5B, if feasible.
- Layer choice: early, middle, late.
- Token position: final token vs function-name token or docstring cue token.
- Surface shift: identifier renaming, formatting changes, comment removal.
- Language shift: Python vs C++ vs Java for semantically equivalent functions.

## 10. Reproducibility Checklist

- Fixed seeds.
- Version-pinned dependencies.
- Config-driven scripts.
- No large artifacts committed to Git.
- Experiment registry updated after every run.
- README updated with final commands and results.
