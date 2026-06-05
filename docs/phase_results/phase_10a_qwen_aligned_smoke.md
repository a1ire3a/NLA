# Phase 10a Results: Qwen-Based Aligned AR/AV Smoke Tests

## Status

Successful.

Qwen-based AR and AV components were implemented and smoke-tested with LoRA using the 0.5B Qwen2.5-Coder model. The goal was to verify implementation correctness, artifact writing, and generation flow before implementing aligned / joint training.

## Design

The Phase 10a direction replaces the DistilBERT/DistilGPT2 debug baseline with Qwen-family components:

- AR: Qwen text encoder / pooled hidden state -> activation vector.
- AV: activation projection -> Qwen causal LM generation.
- Training: LoRA/PEFT by default.
- Alignment: both AR and AV default to explanation-style text via `reference_description`.
- Smoke model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- Final intended model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

## Qwen AR Smoke Test

### Command

```bash
python scripts/train_qwen_ar.py \
  --activation_dir outputs/activations/train_qwen25_coder_15b_l19_ctx512 \
  --validation_activation_dir outputs/activations/validation_qwen25_coder_15b_l19_ctx512 \
  --output_dir outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora \
  --model_name_or_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --text_field reference_description \
  --fallback_text_fields prompt,code \
  --target_transform standardize \
  --limit_train 128 \
  --limit_validation 64 \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --max_length 256 \
  --dtype bfloat16 \
  --seed 42
```

### Result

| Field | Value |
|---|---:|
| Train examples | 128 |
| Validation examples | 64 |
| Target activation dim | 1536 |
| Base model parameters | 499,809,664 |
| Trainable parameters | 5,776,896 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Epochs | 1 |
| Train MSE | 3.072038 |
| Validation FVE | -0.063300 |
| Validation MSE | 0.206061 |
| Status | Success |

### Output Artifacts

```text
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/projection_head.pt
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/tokenizer/
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/qwen_adapter/
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/model.pt
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/training_metrics.csv
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/validation_predictions.pt
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/validation_targets.pt
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/validation_metadata.jsonl
outputs/checkpoints/qwen_ar/smoke_refdesc_qwen05b_lora/train_qwen_ar_manifest.json
```

## Qwen AV Smoke Test

### Command

```bash
python scripts/train_qwen_av.py \
  --activation_dir outputs/activations/train_qwen25_coder_15b_l19_ctx512 \
  --validation_activation_dir outputs/activations/validation_qwen25_coder_15b_l19_ctx512 \
  --output_dir outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora \
  --model_name_or_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --target_text_field reference_description \
  --fallback_text_fields prompt,code \
  --limit_train 128 \
  --limit_validation 64 \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --max_target_length 128 \
  --dtype bfloat16 \
  --seed 42
```

### Result

| Field | Value |
|---|---:|
| Train examples | 128 |
| Validation examples | 64 |
| Base model parameters | 499,809,024 |
| Trainable parameters | 5,776,256 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Epochs | 1 |
| Train loss | 3.429286 |
| Validation loss | 3.137831 |
| Generated validation rows | 64 |
| Status | Success |

### Output Artifacts

```text
outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora/activation_projection.pt
outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora/tokenizer/
outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora/qwen_adapter/
outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora/model.pt
outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora/training_metrics.csv
outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora/validation_generations.jsonl
outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora/train_qwen_av_manifest.json
```

## Qwen AV Generation Smoke Test

### Command

```bash
python scripts/generate_qwen_av_explanations.py \
  --checkpoint_dir outputs/checkpoints/qwen_av/smoke_refdesc_qwen05b_lora \
  --activation_dir outputs/activations/validation_qwen25_coder_15b_l19_ctx512 \
  --output_jsonl outputs/reports/qwen_av/smoke_refdesc_qwen05b_generations.jsonl \
  --limit 20 \
  --batch_size 2 \
  --max_new_tokens 128 \
  --seed 42
```

### Result

Generation completed successfully and wrote:

```text
outputs/reports/qwen_av/smoke_refdesc_qwen05b_generations.jsonl
```

## Interpretation

The smoke tests confirm that the Qwen-based aligned implementation is operational:

1. Qwen AR trains with LoRA and writes all expected artifacts.
2. Qwen AV trains with LoRA and writes all expected artifacts.
3. Qwen AV generation works from saved activation vectors.
4. The 0.5B Qwen smoke setup is viable for debugging and joint-training implementation.
5. The metrics from one epoch on 128 examples are not meant to be a final quality result.

## Decision

Proceed to Phase 10c: implement aligned / joint training.

The next phase should keep using 0.5B Qwen for smoke testing. The 1.5B model should be reserved for the final serious run after the joint/aligned training code is validated.
