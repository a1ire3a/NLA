# Source Usage and Attribution Policy

This project is a small-scale, independent reimplementation of Natural Language Autoencoders (NLA) for an AI4Code setting.

The implementation is inspired by the official NLA repository:

- Official repository: https://github.com/kitft/natural_language_autoencoders
- Article: https://transformer-circuits.pub/2026/nla/index.html
- License of the official repository: Apache-2.0

## Project Positioning

This repository is **not** a fork of the official NLA repository.

The goal is to build a compact and understandable implementation adapted to limited compute and code-language-model experiments. The official repository is used as a reference implementation for architecture, terminology, training stages, and evaluation methodology.

## Clean Reimplementation Rule

By default, this project should use a clean, minimal implementation rather than copying the full official codebase.

Acceptable use:

- Reading the official implementation to understand AV and AR design.
- Reusing high-level architecture ideas.
- Reusing terminology such as AV, AR, activation injection, residual stream, and FVE.
- Citing the official repository and article in documentation.
- Implementing smaller versions of the same ideas in our own code.

Use that requires explicit attribution:

- Copying a function, class, prompt template, config pattern, or substantial code block from the official repository.
- Adapting code with only minor modifications.
- Using official metadata formats or sidecar contracts directly.

If direct code is copied or adapted, the file must include a short note near the top:

```text
Portions of this file are adapted from kitft/natural_language_autoencoders,
licensed under Apache-2.0. See docs/source_usage.md for details.
```

## Why Not Fork the Official Repository?

The official repository is a full training codebase that includes data generation, SFT, RL training, rollout serving, checkpoint conversion, and large-model infrastructure. This project needs a smaller and clearer implementation that can run on a single RTX 3090 Ti.

A clean repository is preferable for this task because it makes the following easier to evaluate:

- Scientific judgment.
- Scope control.
- Reproducibility.
- Understanding of the NLA method.
- Adaptation to AI4Code rather than direct replication of the original setup.

## Required Reference Alignment

Before implementing AV and AR, check the official NLA repository and document how our implementation maps to the original design.

The mapping should cover:

- AV input format.
- Activation injection mechanism.
- AV generation prompt.
- AR input format.
- AR output head.
- Normalization and reconstruction loss.
- FVE or equivalent reconstruction metric.
- Differences caused by our smaller compute budget.

## Citation Note

The final report should cite the NLA article and the official repository as the methodological source of the project.
