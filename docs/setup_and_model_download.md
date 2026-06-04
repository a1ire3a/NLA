# Setup and Model Download Guide

This guide describes the initial environment setup for the NLA-for-code-semantics project.

The preferred environment is a CUDA machine with an NVIDIA RTX 3090 Ti, 24 GB VRAM, and 64 GB system RAM.

## 1. Clone the Repository

```bash
git clone https://github.com/a1ire3a/NLA.git
cd NLA
```

## 2. Create a Python Environment

Recommended Python version: 3.10 or 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch with CUDA

Install the CUDA-enabled PyTorch build that matches the local driver and CUDA runtime.

Example for CUDA 12.4 builds:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If CUDA 12.1 is required instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA availability:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY
```

## 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

## 5. Configure Hugging Face Cache

Large model files should not be stored inside the Git repository.

Recommended local structure:

```bash
mkdir -p ~/hf_cache
export HF_HOME=~/hf_cache
export HF_HUB_CACHE=~/hf_cache/hub
```

For persistent configuration, add the export commands to `~/.bashrc` or `~/.zshrc`.

## 6. Optional: Login to Hugging Face

Most selected models are publicly accessible, but logging in avoids rate limits and supports private or gated datasets if needed.

```bash
huggingface-cli login
```

## 7. Download the Smoke-Test Model

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --local-dir ~/hf_cache/models/Qwen2.5-Coder-0.5B-Instruct \
  --local-dir-use-symlinks False
```

## 8. Download the Main Model

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --local-dir ~/hf_cache/models/Qwen2.5-Coder-1.5B-Instruct \
  --local-dir-use-symlinks False
```

## 9. Run the Feasibility Probe

After implementation, the first script to run will be:

```bash
python scripts/feasibility_probe.py \
  --model_name_or_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --layer_index 16 \
  --max_length 128
```

Expected checks:

- CUDA is available.
- Model and tokenizer load correctly.
- A prompt can be tokenized.
- Hidden states can be extracted.
- The selected layer has the expected shape.
- Final non-padding token activation can be selected.

Then repeat with the main model:

```bash
python scripts/feasibility_probe.py \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 128
```

## 10. Notes on Ollama

Ollama can be useful for informal prompt inspection, but it should not be used for the research pipeline because the project requires direct access to hidden states, `inputs_embeds`, fine-tuning adapters, and activation tensors. The main implementation should use Hugging Face Transformers and PyTorch.
