# Setup and Model Download Guide

This guide describes the initial environment setup for the NLA-for-code-semantics project.

The preferred environment is a CUDA machine with an NVIDIA RTX 3090 Ti, 24 GB VRAM, and 64 GB system RAM.

## Quick setup summary

The project uses **Conda** for environment management so that the Python version is explicit and reproducible.

Recommended Python version: **3.11**.

```bash
conda create -n nla-code python=3.11 -y
conda activate nla-code
python -m pip install --upgrade pip setuptools wheel
```

Then install CUDA-enabled PyTorch and the project requirements.

---

## Manual installation

This section is the full manual setup path for the Ubuntu machine that hosts the RTX 3090 Ti.

The commands assume:

- Ubuntu 22.04 or 24.04.
- NVIDIA driver is already installed.
- `nvidia-smi` works.
- Miniconda or Anaconda is installed.
- CUDA-enabled PyTorch will be installed through pip inside a Conda environment.
- Large artifacts are stored outside Git whenever possible.

### A. Verify the GPU machine

```bash
nvidia-smi
nvcc --version || true
python3 --version
conda --version
```

If `nvidia-smi` does not show the RTX 3090 Ti, fix the NVIDIA driver before continuing.

If `conda --version` fails, install Miniconda first.

### B. Install Miniconda, if needed

Skip this section if Conda is already installed.

```bash
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Close and reopen the terminal, or run:

```bash
source ~/.bashrc
```

Verify:

```bash
conda --version
```

Recommended Conda settings:

```bash
conda config --set auto_activate_base false
```

### C. Install system packages

```bash
sudo apt update
sudo apt install -y \
  git \
  git-lfs \
  curl \
  wget \
  build-essential \
  unzip \
  htop

git lfs install
```

### D. Clone the project

```bash
mkdir -p ~/research
cd ~/research

git clone https://github.com/a1ire3a/NLA.git
cd NLA
```

### E. Create and activate the Conda environment

Use Python 3.11 as the default project version:

```bash
conda create -n nla-code python=3.11 -y
conda activate nla-code
python -m pip install --upgrade pip setuptools wheel
```

If you need Python 3.10 instead, create a separate environment:

```bash
conda create -n nla-code-py310 python=3.10 -y
conda activate nla-code-py310
python -m pip install --upgrade pip setuptools wheel
```

For the rest of this guide, assume the active environment is:

```bash
conda activate nla-code
```

### F. Install CUDA-enabled PyTorch

Use the CUDA wheel that matches your installed NVIDIA driver. For most recent NVIDIA drivers, start with CUDA 12.4 wheels:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If CUDA 12.4 wheels do not work on your machine, use CUDA 12.1 wheels instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch CUDA access:

```bash
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('device name:', torch.cuda.get_device_name(0))
    print('allocated GB:', torch.cuda.memory_allocated(0) / 1024**3)
PY
```

### G. Install project dependencies

Make sure you are inside the repository and the Conda environment is active:

```bash
cd ~/research/NLA
conda activate nla-code
pip install -r requirements.txt
```

If `flash-attn` is later needed, install it separately only after the basic pipeline works. Do not add it to the first setup path.

### H. Configure Hugging Face cache and local artifact folders

Large model files and datasets should not be stored inside the Git repository.

```bash
mkdir -p ~/hf_cache/hub
mkdir -p ~/hf_cache/models
mkdir -p ~/hf_cache/datasets
mkdir -p ~/research/NLA/data/raw
mkdir -p ~/research/NLA/data/interim
mkdir -p ~/research/NLA/data/processed
mkdir -p ~/research/NLA/outputs/activations
mkdir -p ~/research/NLA/outputs/checkpoints
mkdir -p ~/research/NLA/outputs/figures
mkdir -p ~/research/NLA/outputs/reports

export HF_HOME=~/hf_cache
export HF_HUB_CACHE=~/hf_cache/hub
export HF_DATASETS_CACHE=~/hf_cache/datasets
```

Optional persistent setup:

```bash
cat >> ~/.bashrc <<'BASHRC'
export HF_HOME=~/hf_cache
export HF_HUB_CACHE=~/hf_cache/hub
export HF_DATASETS_CACHE=~/hf_cache/datasets
BASHRC
```

Reload if needed:

```bash
source ~/.bashrc
conda activate nla-code
```

### I. Login to Hugging Face

```bash
huggingface-cli login
```

This is optional for public models and datasets, but recommended to avoid rate limits.

### J. Download the selected Code LLMs

Official model pages:

- `Qwen/Qwen2.5-Coder-0.5B-Instruct`: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct

Download the smoke-test model:

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --local-dir ~/hf_cache/models/Qwen2.5-Coder-0.5B-Instruct \
  --local-dir-use-symlinks False
```

Download the main model:

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --local-dir ~/hf_cache/models/Qwen2.5-Coder-1.5B-Instruct \
  --local-dir-use-symlinks False
```

Quick model-loading check:

```bash
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
print('loaded:', model_name)
print('num layers:', len(model.model.layers))
print('hidden size:', model.config.hidden_size)
PY
```

### K. Download the primary training-style dataset: CodeSearchNet / CodeXGLUE

Official references:

- CodeXGLUE repository: https://github.com/microsoft/CodeXGLUE
- CodeXGLUE paper and benchmark page are linked from the repository.
- CodeSearchNet is the source dataset used by CodeXGLUE for code summarization and code search.

Keep a local copy of the CodeXGLUE repository for reference scripts and documentation:

```bash
mkdir -p ~/research/external
cd ~/research/external

git clone https://github.com/microsoft/CodeXGLUE.git
```

Return to the project:

```bash
cd ~/research/NLA
conda activate nla-code
```

Primary dataset loading path through Hugging Face `datasets`:

```bash
python - <<'PY'
from datasets import load_dataset

cache_dir = 'data/raw/hf_datasets'
output_dir = 'data/raw/code_search_net_python'

print('Loading CodeSearchNet Python split...')
ds = load_dataset(
    'code_search_net',
    'python',
    cache_dir=cache_dir,
    trust_remote_code=True,
)
print(ds)
ds.save_to_disk(output_dir)
print(f'Saved to {output_dir}')
PY
```

If this fails because of dataset-script restrictions or upstream changes, use the CodeXGLUE clone as the fallback reference and implement the downloader in `scripts/prepare_dataset.py` based on the current CodeXGLUE directory layout.

### L. Download the controlled multilingual dataset: HumanEval-X

Official references:

- Hugging Face dataset page: https://huggingface.co/datasets/THUDM/humaneval-x
- CodeGeeX repository: https://github.com/THUDM/CodeGeeX
- Redirected active repository may appear as: https://github.com/zai-org/CodeGeeX

Keep a local copy of the CodeGeeX repository for benchmark scripts and data references:

```bash
mkdir -p ~/research/external
cd ~/research/external

git clone https://github.com/THUDM/CodeGeeX.git || git clone https://github.com/zai-org/CodeGeeX.git
```

Return to the project:

```bash
cd ~/research/NLA
conda activate nla-code
```

Download HumanEval-X through Hugging Face `datasets` for the languages used in this project:

```bash
python - <<'PY'
from datasets import load_dataset

cache_dir = 'data/raw/hf_datasets'
langs = ['python', 'cpp', 'java']

for lang in langs:
    print(f'Loading HumanEval-X: {lang}')
    try:
        ds = load_dataset(
            'THUDM/humaneval-x',
            lang,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as first_error:
        print(f'THUDM alias failed for {lang}: {first_error}')
        print(f'Trying zai-org/humaneval-x for {lang}')
        ds = load_dataset(
            'zai-org/humaneval-x',
            lang,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    out = f'data/raw/humaneval_x_{lang}'
    print(ds)
    ds.save_to_disk(out)
    print(f'Saved to {out}')
PY
```

Optional: download the dataset repository files directly with `huggingface-cli`:

```bash
huggingface-cli download THUDM/humaneval-x \
  --repo-type dataset \
  --local-dir data/raw/humaneval-x-hf \
  --local-dir-use-symlinks False || \
huggingface-cli download zai-org/humaneval-x \
  --repo-type dataset \
  --local-dir data/raw/humaneval-x-hf \
  --local-dir-use-symlinks False
```

### M. Optional small Python-only sanity dataset: OpenAI HumanEval

This is not the main dataset, but it is useful as a tiny sanity check because it has only 164 test rows.

Official dataset page:

- https://huggingface.co/datasets/openai/openai_humaneval

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset('openai/openai_humaneval', cache_dir='data/raw/hf_datasets')
print(ds)
ds.save_to_disk('data/raw/openai_humaneval')
print('Saved to data/raw/openai_humaneval')
PY
```

### N. Verify that local datasets are readable

```bash
python - <<'PY'
from datasets import load_from_disk

paths = [
    'data/raw/code_search_net_python',
    'data/raw/humaneval_x_python',
    'data/raw/humaneval_x_cpp',
    'data/raw/humaneval_x_java',
]

for path in paths:
    try:
        ds = load_from_disk(path)
        print('\nPATH:', path)
        print(ds)
        split = 'train' if 'train' in ds else 'test'
        print('first keys:', list(ds[split][0].keys()))
    except Exception as exc:
        print('\nFAILED:', path, exc)
PY
```

### O. Run the initial repository health checks

```bash
cd ~/research/NLA
conda activate nla-code
pytest -q
python -m compileall src scripts
```

### P. Expected local directory state

After manual installation, the important local paths should look like this:

```text
~/research/NLA/
├── data/
│   └── raw/
│       ├── code_search_net_python/
│       ├── humaneval_x_python/
│       ├── humaneval_x_cpp/
│       ├── humaneval_x_java/
│       └── openai_humaneval/              # optional
├── outputs/
│   ├── activations/
│   ├── checkpoints/
│   ├── figures/
│   └── reports/
└── ...

~/hf_cache/
├── hub/
├── datasets/
└── models/
    ├── Qwen2.5-Coder-0.5B-Instruct/
    └── Qwen2.5-Coder-1.5B-Instruct/

~/research/external/
├── CodeXGLUE/
└── CodeGeeX/
```

### Q. What not to commit

Do not commit:

- Hugging Face model weights.
- Raw datasets.
- Processed large datasets.
- Extracted activations.
- Checkpoints.
- Generated reports and figures before they are finalized.

Only commit code, configs, small CSV logs, documentation, and selected lightweight final artifacts.
