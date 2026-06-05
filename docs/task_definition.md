# Task Definition

## Project objective

Build a compact and reproducible Natural Language Autoencoder (NLA) pipeline for a small open-source code language model. The system should translate internal model activations into natural-language explanations and reconstruct the original activations from those explanations.

The project is not intended to reproduce the full scale of the original NLA work. It is a focused feasibility study designed for limited compute and careful scientific analysis.

## Core research question

> Can a simplified NLA recover meaningful information from the residual-stream activations of a small code language model, and do the generated explanations reflect code semantics rather than only superficial syntax?

## Target task

The target task is **function-level code understanding**.

For each example, the target model receives a prompt containing a code function and is positioned immediately before generating an explanation of the function. The activation at the final non-padding prompt token is extracted from a selected transformer layer.

Example prompt shape:

```text
Read the following function and prepare to explain what it does.

<code>
...
</code>

Explanation:
```

The extracted activation becomes the input to the Activation Verbalizer (AV). The AV produces a short textual explanation, and the Activation Reconstructor (AR) attempts to reconstruct the original activation from that text.

## Initial experimental conditions

### Target models

- Primary model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Smoke-test model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`

### Activation definition

- Representation: residual-stream hidden state
- Token position: final non-padding token of the prompt
- Layer: approximately two-thirds through the model, finalized after feasibility probing

### Data domains

- Training and validation: Python function-level code examples
- Controlled test conditions:
  - In-domain Python functions
  - Identifier-renamed Python functions
  - Formatting and comment perturbations
  - Same-semantics functions across Python, C++, and Java

### Processed dataset schema

Dataset preparation writes JSONL rows with the following fields:

- `example_id`
- `source_dataset`
- `source_split`
- `split`
- `language`
- `task_family`
- `code`
- `prompt`
- `reference_description`
- `transformation_type`
- `paired_example_id`
- `metadata`

The prompt is constructed with `code_explanation_prompt_v1(code)`. The
`transformation_type` field records whether the row is original code, an
identifier-renamed surface shift, a formatting/comment-only shift, or a
language-shift example.

## Required outputs

The project should produce:

1. A reproducible dataset preparation pipeline
2. A reproducible activation extraction pipeline
3. Baseline reconstruction methods
4. An Activation Reconstructor (text to vector)
5. An Activation Verbalizer (vector to text)
6. A complete NLA loop (vector to text to vector)
7. Quantitative reconstruction evaluation using FVE
8. Qualitative analysis of explanations and failure modes
9. Controlled robustness and generalization experiments
10. A concise final report and experiment registry

## Primary metric

The main reconstruction metric is Fraction of Variance Explained (FVE).

```text
FVE = 1 - reconstruction_error / baseline_variance
```

The exact implementation must be documented and tested before model training.

## Scientific success criteria

The project is successful if it can provide a clear and reproducible answer to the following questions:

- Does the NLA reconstruct activations better than simple baselines?
- Are the generated explanations understandable and related to code behavior?
- Does reconstruction quality change under surface-level code transformations?
- Do explanations remain semantically consistent across programming languages?
- Which failure modes limit the method on small models and limited compute?

## Out of scope for the initial study

- Reproducing large-scale NLA reinforcement learning experiments
- Training large 7B+ models from scratch
- Interpreting every layer or every token position
- Claiming that generated explanations are exact descriptions of model reasoning
- Treating FVE alone as proof of semantic faithfulness
