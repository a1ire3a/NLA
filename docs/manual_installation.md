# Manual Installation and Reproduction Guide

This document summarizes how to reproduce the project environment and the main experiment pipeline on an Ubuntu CUDA workstation.

## Hardware used

The reported experiments were run on a single workstation with:

- NVIDIA RTX 3090 Ti 24GB GPU
- 64-core CPU
- 128GB RAM

The full hardware capacity was not always saturated, but this is the reproducibility envelope for the reported runs.

## 1. Clone the repository

```bash
git clone https://github.com/a1ire3a/NLA.git
cd NLA
```

## 2. Create the Conda environment

```bash
conda create -n nla python=3.11 -y
conda activate nla
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install a PyTorch build compatible with the local driver. The experiments used a CUDA-enabled PyTorch build on an RTX 3090 Ti.

Check CUDA availability:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 3. Download models

The main model is:

```text
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

The smoke-test model is:

```text
Qwen/Qwen2.5-Coder-0.5B-Instruct
```

The scripts use Hugging Face model names directly, so models are downloaded automatically unless already cached.

Optional pre-download:

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct
```

## 4. Prepare local data directories

Large raw and processed data files are intentionally not committed.

Expected local layout:

```text
data/raw/code_search_net_python/
data/raw/humaneval_x_python/
data/raw/humaneval_x_cpp/
data/raw/humaneval_x_java/
data/processed/
outputs/
```

The dataset preparation script expects CodeSearchNet-style Python data and HumanEval-X-style language-shift data in those folders.

## 5. Prepare processed datasets

```bash
python scripts/prepare_dataset.py \
  --codesearchnet_path data/raw/code_search_net_python \
  --humaneval_x_python_path data/raw/humaneval_x_python \
  --humaneval_x_cpp_path data/raw/humaneval_x_cpp \
  --humaneval_x_java_path data/raw/humaneval_x_java \
  --output_dir data/processed \
  --pilot_size 100 \
  --train_size 5000 \
  --validation_size 500 \
  --test_size 500 \
  --seed 42
```

Expected processed files:

```text
data/processed/pilot_100.jsonl
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test_indomain.jsonl
data/processed/test_surface_shift.jsonl
data/processed/test_language_shift.jsonl
```

## 6. Extract activations

Main train and validation activations:

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/train.jsonl \
  --output_dir outputs/activations/train_qwen25_coder_15b_l19_ctx512 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 512 \
  --batch_size 8 \
  --dtype bfloat16 \
  --seed 42
```

```bash
python scripts/extract_activations.py \
  --input_jsonl data/processed/validation.jsonl \
  --output_dir outputs/activations/validation_qwen25_coder_15b_l19_ctx512 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --layer_index 19 \
  --max_length 512 \
  --batch_size 8 \
  --dtype bfloat16 \
  --seed 42
```

Test splits use the same script with the corresponding `data/processed/test_*.jsonl` inputs.

## 7. Run the final aligned Qwen training

```bash
python scripts/train_qwen_joint_nla.py \
  --activation_dir outputs/activations/train_qwen25_coder_15b_l19_ctx512 \
  --validation_activation_dir outputs/activations/validation_qwen25_coder_15b_l19_ctx512 \
  --output_dir outputs/checkpoints/qwen_joint/final_qwen15b_full_e20 \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --target_text_field reference_description \
  --fallback_text_fields prompt,code \
  --target_transform standardize \
  --epochs 20 \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate_av 1e-4 \
  --learning_rate_ar 1e-4 \
  --max_target_length 128 \
  --max_ar_length 256 \
  --max_new_tokens 64 \
  --dtype bfloat16 \
  --seed 42
```

## 8. Run reward-driven AV optimization

```bash
python scripts/train_qwen_av_reward_rl.py \
  --activation_dir outputs/activations/train_qwen25_coder_15b_l19_ctx512 \
  --validation_activation_dir outputs/activations/validation_qwen25_coder_15b_l19_ctx512 \
  --joint_checkpoint_dir outputs/checkpoints/qwen_joint/final_qwen15b_full_e20 \
  --output_dir outputs/checkpoints/qwen_rl/final_qwen15b_av_reward_rl \
  --epochs 3 \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate_av 5e-5 \
  --max_new_tokens 64 \
  --max_ar_length 256 \
  --temperature 0.7 \
  --top_p 0.95 \
  --reward_normalization batch_zscore \
  --kl_weight 0.01 \
  --dtype bfloat16 \
  --seed 42 \
  --eval_every_epoch
```

## 9. Evaluate the final reward-RL system

Example for the in-domain test split:

```bash
python scripts/run_qwen_nla_loop.py \
  --activation_dir outputs/activations/test_indomain_qwen25_coder_15b_l19_ctx512 \
  --rl_av_checkpoint_dir outputs/checkpoints/qwen_rl/final_qwen15b_av_reward_rl \
  --joint_checkpoint_dir outputs/checkpoints/qwen_joint/final_qwen15b_full_e20 \
  --output_dir outputs/reports/qwen_nla_loop \
  --run_name test_indomain_qwen15b_av_reward_rl \
  --batch_size 2 \
  --max_new_tokens 64 \
  --seed 42
```

Repeat with:

```text
outputs/activations/test_surface_shift_qwen25_coder_15b_l19_ctx512
outputs/activations/test_language_shift_qwen25_coder_15b_l19_ctx512
```

## 10. Where results are documented

- Final README summary: `README.md`
- Central narrative log: `docs/research_log.md`
- Phase reports: `docs/phase_results/`
- CSV experiment registries: `experiments/`

Large raw artifacts and checkpoints are intentionally kept outside Git.
