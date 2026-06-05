# Phase 1 Results: Feasibility Probe

## Status

**Accepted / successful.**

Both selected Qwen2.5-Coder models loaded successfully on the CUDA machine, returned hidden states with the expected shapes, allowed final-token activation extraction, and passed the `inputs_embeds` compatibility check.

## Execution Environment

- Host prompt: `ajavaheri@DeepFake`
- Python: `3.11.4`
- OS/platform reported by Python: `Linux-3.10.0-1160.el7.x86_64-x86_64-with-glibc2.31`
- PyTorch: `2.6.0+cu124`
- PyTorch CUDA version: `12.4`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 3090`
- GPU capability: `(8, 6)`

Note: the project target was described as RTX 3090 Ti, but the runtime reported `NVIDIA GeForce RTX 3090`. The probe still succeeded and VRAM usage is well within 24 GB.

## Commands Run

### Smoke-test model

```bash
python scripts/feasibility_probe.py \
  --model_name_or_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --layer_index 16 \
  --max_length 128 \
  --dtype bfloat16
```

### Main model

```bash
python scripts/feasibility_probe.py \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 128 \
  --dtype bfloat16
```

## Smoke-Test Model Result

Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`

| Field | Value |
|---|---:|
| Tokenizer class | `Qwen2TokenizerFast` |
| Vocabulary size | `151643` |
| Model class | `Qwen2ForCausalLM` |
| Hidden size | `896` |
| Model layers | `24` |
| Selected hidden-state tuple index | `16` |
| Number of hidden-state tensors | `25` |
| Input shape | `(1, 57)` |
| Attention mask shape | `(1, 57)` |
| Non-padding tokens | `57` |
| Selected hidden-state shape | `(1, 57, 896)` |
| Final non-padding token index | `[56]` |
| Activation shape | `(1, 896)` |
| Activation dtype | `torch.bfloat16` |
| Activation device | `cuda:0` |
| Activation mean | `0.071852` |
| Activation std | `0.884893` |
| Activation min | `-11.625000` |
| Activation max | `12.500000` |
| Activation L2 norm | `26.574921` |
| Initial VRAM | `0.00 GB allocated / 0.00 GB reserved` |
| After model load | `0.92 GB allocated / 0.93 GB reserved` |
| After hidden-state forward | `0.95 GB allocated / 0.95 GB reserved` |
| After inputs_embeds forward | `0.97 GB allocated / 0.97 GB reserved` |
| inputs_embeds check | Passed |
| Final status | Success |

## Main Model Result

Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`

| Field | Value |
|---|---:|
| Tokenizer class | `Qwen2TokenizerFast` |
| Vocabulary size | `151643` |
| Model class | `Qwen2ForCausalLM` |
| Hidden size | `1536` |
| Model layers | `28` |
| Selected hidden-state tuple index | `19` |
| Number of hidden-state tensors | `29` |
| Input shape | `(1, 57)` |
| Attention mask shape | `(1, 57)` |
| Non-padding tokens | `57` |
| Selected hidden-state shape | `(1, 57, 1536)` |
| Final non-padding token index | `[56]` |
| Activation shape | `(1, 1536)` |
| Activation dtype | `torch.bfloat16` |
| Activation device | `cuda:0` |
| Activation mean | `0.077203` |
| Activation std | `1.323934` |
| Activation min | `-20.750000` |
| Activation max | `18.375000` |
| Activation L2 norm | `51.975567` |
| Initial VRAM | `0.00 GB allocated / 0.00 GB reserved` |
| After model load | `2.88 GB allocated / 2.93 GB reserved` |
| After hidden-state forward | `2.90 GB allocated / 2.93 GB reserved` |
| After inputs_embeds forward | `2.93 GB allocated / 2.93 GB reserved` |
| inputs_embeds check | Passed |
| Final status | Success |

## Interpretation

Phase 1 validates the core technical assumptions needed before implementing dataset preparation and activation extraction:

1. The CUDA environment can load both selected models.
2. The smoke-test model is cheap enough for iterative debugging.
3. The main 1.5B model is feasible on the available GPU.
4. Hidden states are returned with expected dimensionality.
5. Final non-padding token activation extraction works.
6. `inputs_embeds` forward compatibility works, which is important for the later AV design based on activation injection.

## Notes for Later Phases

- The runtime emitted this warning: `` `torch_dtype` is deprecated! Use `dtype` instead! ``. This is not a blocker for Phase 1, but the model-loading helper should be updated in a future cleanup step.
- The reported GPU name is RTX 3090, not RTX 3090 Ti. This does not change the immediate feasibility result.
- The main model uses only about 3 GB VRAM for this single-sample probe, leaving enough room for pilot activation extraction and small adapter-based experiments.

## Decision

Proceed to **Phase 2: Dataset Preparation**.
