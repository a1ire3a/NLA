# Official NLA Reference Mapping

This document records how the local implementation maps to the official Natural Language Autoencoder design.

Official reference repository:

- https://github.com/kitft/natural_language_autoencoders

Official article:

- https://transformer-circuits.pub/2026/nla/index.html

## Mapping Table

| Component | Official NLA design | Local implementation decision | Status |
|---|---|---|---|
| Target model | Frozen language model whose activations are interpreted | Frozen Qwen2.5-Coder model | Planned |
| Target activation | Residual-stream hidden state from a selected layer/token | Final non-padding prompt token, layer around 2/3 depth | Planned |
| AV | Activation Verbalizer, vector to natural-language text | Minimal activation-to-text module, aligned with official injection concept | Planned |
| Activation injection | Activation inserted as an embedding-like special token in the AV prompt | To be implemented using `inputs_embeds` in Hugging Face Transformers | Planned |
| AV training | SFT initialization followed by RL in the full setup | Start with supervised/pseudo-supervised and small-scale reconstruction-driven loop; RL optional/out of initial scope | Planned |
| AR | Activation Reconstructor, text to vector | Minimal text encoder / truncated LM with linear output head | Planned |
| AR output | Reconstructed activation vector | Same activation dimensionality as target hidden state | Planned |
| Reconstruction loss | MSE or related vector reconstruction objective | MSE plus FVE evaluation | Planned |
| Primary metric | Fraction of Variance Explained / reconstruction quality | FVE against baselines | Planned |
| Scope | Large-scale multi-model NLA training | Single-GPU small-model AI4Code feasibility study | Planned |

## Implementation Checkpoints

Before merging any implementation of AV or AR, verify:

- [ ] It can run on the smoke-test model.
- [ ] It does not require the full official training infrastructure.
- [ ] It documents any direct code adaptation.
- [ ] It has a small unit test or smoke test.
- [ ] It logs enough metadata for reproducibility.

## Open Design Questions

1. Should AV and AR share the same base model family as the target model?
2. Should AR be a truncated causal LM or a smaller encoder-style model?
3. Should the first AV training phase use synthetic explanations, original code docstrings, or direct reconstruction feedback?
4. How many explanation tokens are enough for the first experiment?
5. Should activation vectors be normalized before injection and reconstruction?

These questions should be answered gradually through feasibility probes and pilot experiments.
