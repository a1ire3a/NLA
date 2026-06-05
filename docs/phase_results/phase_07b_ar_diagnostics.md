# Phase 7b Results: AR Diagnostics and Improvement

## Status

**Pilot diagnostics successful.**

The AR diagnostics confirmed that target normalization matters substantially. The first positive AR validation FVE was obtained with standardized activation targets and frozen DistilBERT representations.

This result is encouraging, but it is still based on the 100-example pilot artifact. The next step should be scaling activation extraction to train/validation before drawing strong conclusions.

## Input Summary File

The diagnostic results were provided as:

```text
ar_phase5b_summary.csv
```

The CSV contained four AR runs with the following columns:

- `run_dir`
- `target_transform`
- `text_field`
- `freeze_text_model`
- `best_epoch`
- `validation_fve`
- `validation_mse`
- `validation_rmse`
- `validation_cosine_mean`
- `validation_train_mean_baseline_fve`
- `validation_train_mean_baseline_mse`
- `beats_validation_train_mean_baseline`

## Results

| Run | Text field | Frozen text model | Target transform | Best epoch | Validation FVE | Validation MSE | Validation RMSE | Validation cosine | Train-mean baseline FVE | Beats train-mean baseline? |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `refdesc_distilbert_center` | `reference_description` | yes | center | 8 | -0.272226 | 0.017328 | 0.131637 | 0.994898 | -0.059908 | no |
| `refdesc_distilbert_standardize` | `reference_description` | yes | standardize | 13 | 0.056828 | 0.012846 | 0.113342 | 0.996221 | -0.059908 | yes |
| `code_distilbert_center` | `code` | yes | center | 8 | -0.032409 | 0.014062 | 0.118583 | 0.995905 | -0.059908 | yes |
| `refdesc_distilbert_center_unfrozen` | `reference_description` | no | center | 10 | -0.099820 | 0.014980 | 0.122393 | 0.995556 | -0.059908 | no |

## Interpretation

The main takeaways are:

1. **Standardization fixed much of the scale/centering issue.**  
   The `reference_description + standardize + frozen DistilBERT` run achieved the best validation FVE: `0.056828`.

2. **The first positive validation FVE was achieved.**  
   This means the AR can reconstruct slightly better than the mean-style reference on this pilot split.

3. **Code text is competitive.**  
   The `code + center + frozen DistilBERT` run had FVE `-0.032409`, but it still beat the train-mean baseline FVE of `-0.059908`. This suggests that source code text may contain useful information for activation reconstruction.

4. **Unfreezing DistilBERT did not help on the 100-example pilot.**  
   The unfrozen run had FVE `-0.099820`, which is worse than the centered frozen reference-description run. With only 80 training examples, unfreezing likely overfits or destabilizes training.

5. **High cosine similarity remains consistent across improved runs.**  
   All diagnostic runs reached cosine means around `0.995` to `0.996`, suggesting that direction reconstruction is easier than full variance reconstruction.

## Comparison to Phase 7a Baseline

The earlier raw-target AR baseline had:

- Validation FVE: `-5.110613`
- Validation RMSE: `0.288495`
- Validation cosine mean: `0.979651`

The best diagnostic run improved to:

- Validation FVE: `0.056828`
- Validation RMSE: `0.113342`
- Validation cosine mean: `0.996221`

This is a large improvement and confirms that target transformation should be part of the AR training protocol.

## Decision

Do not proceed to AV yet.

The correct next step is:

1. Extract main-model train activations with `max_length=512`.
2. Extract main-model validation activations with `max_length=512`.
3. Run baseline evaluation on those artifacts.
4. Train AR on the larger activation set using `target_transform=standardize`.
5. Compare at least `reference_description`, `code`, and optionally `prompt` as AR text sources.

## Recommended Next AR Configuration

Use the following as the default AR configuration for the larger train/validation run:

```text
text_model_name_or_path = distilbert-base-uncased
text_field = reference_description
fallback_text_fields = prompt,code
freeze_text_model = true
target_transform = standardize
epochs = 20 to 40
batch_size = 16
learning_rate = 1e-3
max_length = 256
```

Optional comparison:

```text
text_field = code
target_transform = center or standardize
freeze_text_model = true
```
