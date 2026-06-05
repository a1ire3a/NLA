# Phase 5 Results: Activation Reconstructor Baseline

## Status

**Technically successful, scientifically not yet sufficient.**

The first Activation Reconstructor (AR) training run completed successfully and produced checkpoint and validation artifacts. However, the validation FVE remains negative, meaning this first AR baseline does not yet outperform the mean activation baseline.

## Code Review Summary

The Phase 5 implementation adds a lightweight text-to-activation reconstructor:

- `TextActivationReconstructor` in `src/nla_code_interp/models.py`.
- Attention-mask-aware mean pooling over a Hugging Face text model.
- Optional frozen or trainable text model.
- Linear or MLP projection head to the target activation dimension.
- AR training script in `scripts/train_ar.py`.
- Deterministic train/validation split for pilot artifacts.
- Validation metrics using the existing reconstruction metric utilities.

The first run used:

- Text model: `distilbert-base-uncased`
- Text field: `reference_description`
- Fallback fields: `prompt,code`
- Text encoder: frozen
- Projection head: trainable
- Activation artifact: `pilot_100_qwen25_coder_15b_l19_ctx512`
- Train/validation split: 80 / 20

## Command Run

```bash
python scripts/train_ar.py \
  --activation_dir outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx512 \
  --output_dir outputs/checkpoints/ar/pilot_100_qwen25_coder_15b_l19_ctx512_refdesc_distilbert_frozen \
  --text_model_name_or_path distilbert-base-uncased \
  --text_field reference_description \
  --fallback_text_fields prompt,code \
  --freeze_text_model \
  --epochs 20 \
  --batch_size 16 \
  --learning_rate 1e-3 \
  --max_length 256 \
  --seed 42
```

## Training Setup

| Field | Value |
|---|---:|
| Activation shape | `(100, 1536)` |
| Activation dtype | `torch.float32` |
| Text model | `distilbert-base-uncased` |
| Tokenizer | `DistilBertTokenizerFast` |
| Pooling | `mean` |
| Text hidden dim | `768` |
| Activation dim | `1536` |
| Text model frozen | `True` |
| Trainable parameters | `1,181,184` |
| Epochs | `20` |
| Batch size | `16` |
| Learning rate | `0.001` |
| Train examples | `80` |
| Validation examples | `20` |

## Final Result

| Metric | Value |
|---|---:|
| Best epoch | `20` |
| Best validation FVE | `-5.110613` |
| Best validation RMSE | `0.288495` |
| Best validation cosine mean | `0.979651` |

The model improved steadily across epochs:

- Validation FVE improved from `-72.555458` at epoch 1 to `-5.110613` at epoch 20.
- Validation RMSE improved from `1.000928` to `0.288495`.
- Validation cosine mean improved from `0.667192` to `0.979651`.

## Output Files

The following generated outputs were written locally and are intentionally not committed because `outputs/` is ignored:

```text
outputs/checkpoints/ar/pilot_100_qwen25_coder_15b_l19_ctx512_refdesc_distilbert_frozen/model.pt
outputs/checkpoints/ar/pilot_100_qwen25_coder_15b_l19_ctx512_refdesc_distilbert_frozen/training_metrics.csv
outputs/checkpoints/ar/pilot_100_qwen25_coder_15b_l19_ctx512_refdesc_distilbert_frozen/validation_predictions.pt
outputs/checkpoints/ar/pilot_100_qwen25_coder_15b_l19_ctx512_refdesc_distilbert_frozen/validation_targets.pt
outputs/checkpoints/ar/pilot_100_qwen25_coder_15b_l19_ctx512_refdesc_distilbert_frozen/validation_metadata.jsonl
outputs/checkpoints/ar/pilot_100_qwen25_coder_15b_l19_ctx512_refdesc_distilbert_frozen/train_ar_manifest.json
```

## Interpretation

This run is useful but not yet strong enough for the full NLA loop.

Important observations:

1. The training pipeline works end-to-end.
2. The projection head can learn a high-cosine approximation of the target activation direction.
3. Validation FVE remains negative, so reconstruction is still worse than the mean activation baseline.
4. The high cosine but negative FVE suggests a possible scale/centering problem: the model may learn activation direction better than activation magnitude or per-dimension variance.
5. The frozen DistilBERT + reference-description setup may be too weak or too data-limited for reconstructing residual-stream activations.

## Comparison to Phase 4 Baselines

For `pilot_100_qwen25_coder_15b_l19_ctx512`, Phase 4 showed:

- Mean baseline FVE: `0.000000`
- Zero baseline FVE: `-51.401844`
- Shuffled baseline FVE: `-1.014737`

The current AR result:

- AR validation FVE: `-5.110613`

Therefore:

- AR is much better than the zero baseline.
- AR is worse than the shuffled baseline and mean baseline.
- AR should be improved before proceeding to the AV-generated-text bottleneck.

## Decision

Proceed to **Phase 5b: AR Diagnostics and Improvement** before implementing AV.

The next phase should add target normalization, residual prediction, text-field comparisons, and more direct baseline comparisons on the same validation split.
