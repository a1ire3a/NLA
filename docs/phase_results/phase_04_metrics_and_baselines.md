# Phase 4 Results: Metrics and Baselines

## Status

**Accepted / successful.**

The baseline evaluation pipeline successfully loaded Phase 3 activation artifacts, validated their metadata and tensor shapes, computed FVE and auxiliary reconstruction metrics, and wrote baseline reports.

## Code Review Summary

The implemented pipeline in `scripts/run_evaluation.py`:

- Loads `activations.pt`, `metadata.jsonl`, and `manifest.json` from a Phase 3 artifact directory.
- Validates tensor dimensionality, metadata row count, manifest `num_examples`, manifest `activation_dim`, and sequential activation indices.
- Supports an optional reference activation directory for mean baselines.
- Builds three baselines:
  - mean reconstruction
  - zero reconstruction
  - shuffled reconstruction
- Computes aggregate metrics:
  - FVE
  - MSE
  - RMSE
  - mean L2 error
  - cosine similarity summary
- Writes CSV, JSON, per-example JSONL, and evaluation manifest outputs.

The metric implementation in `src/nla_code_interp/metrics.py` includes:

- `fraction_variance_explained`
- `per_example_squared_error`
- `per_example_l2_error`
- `per_example_cosine_similarity`
- `cosine_similarity_summary`
- mean, zero, and shuffled reconstruction baselines
- `summarize_reconstruction`

## Commands Run

### Main model, context 512

```bash
python scripts/run_evaluation.py \
  --activation_dir outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx512 \
  --output_dir outputs/reports/baselines \
  --run_name pilot_100_qwen25_coder_15b_l19_ctx512 \
  --seed 42
```

### Main model, context 1024

```bash
python scripts/run_evaluation.py \
  --activation_dir outputs/activations/pilot_100_qwen25_coder_15b_l19_ctx1024 \
  --output_dir outputs/reports/baselines \
  --run_name pilot_100_qwen25_coder_15b_l19_ctx1024 \
  --seed 42
```

### Small model, context 128

```bash
python scripts/run_evaluation.py \
  --activation_dir outputs/activations/pilot_100_qwen25_coder_05b_l16 \
  --output_dir outputs/reports/baselines \
  --run_name pilot_100_qwen25_coder_05b_l16 \
  --seed 42
```

## Output Files

The following reports were generated locally and intentionally not committed because `outputs/` is ignored:

```text
outputs/reports/baselines/pilot_100_qwen25_coder_15b_l19_ctx512_baseline_metrics.csv
outputs/reports/baselines/pilot_100_qwen25_coder_15b_l19_ctx512_baseline_metrics.json
outputs/reports/baselines/pilot_100_qwen25_coder_15b_l19_ctx512_per_example_errors.jsonl
outputs/reports/baselines/pilot_100_qwen25_coder_15b_l19_ctx512_evaluation_manifest.json

outputs/reports/baselines/pilot_100_qwen25_coder_15b_l19_ctx1024_baseline_metrics.csv
outputs/reports/baselines/pilot_100_qwen25_coder_15b_l19_ctx1024_baseline_metrics.json
outputs/reports/baselines/pilot_100_qwen25_coder_15b_l19_ctx1024_per_example_errors.jsonl
outputs/reports/baselines/pilot_100_qwen25_coder_15b_l19_ctx1024_evaluation_manifest.json

outputs/reports/baselines/pilot_100_qwen25_coder_05b_l16_baseline_metrics.csv
outputs/reports/baselines/pilot_100_qwen25_coder_05b_l16_baseline_metrics.json
outputs/reports/baselines/pilot_100_qwen25_coder_05b_l16_per_example_errors.jsonl
outputs/reports/baselines/pilot_100_qwen25_coder_05b_l16_evaluation_manifest.json
```

## Aggregate Results

| Artifact | Baseline | FVE | MSE |
|---|---|---:|---:|
| `pilot_100_qwen25_coder_15b_l19_ctx512` | mean | 0.000000 | 0.031744 |
| `pilot_100_qwen25_coder_15b_l19_ctx512` | zero | -51.401844 | 1.663463 |
| `pilot_100_qwen25_coder_15b_l19_ctx512` | shuffled | -1.014737 | 0.063957 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | mean | 0.000000 | 0.017681 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | zero | -93.230362 | 1.666052 |
| `pilot_100_qwen25_coder_15b_l19_ctx1024` | shuffled | -1.001524 | 0.035388 |
| `pilot_100_qwen25_coder_05b_l16` | mean | 0.000000 | 0.452157 |
| `pilot_100_qwen25_coder_05b_l16` | zero | -0.668432 | 0.754392 |
| `pilot_100_qwen25_coder_05b_l16` | shuffled | -0.920304 | 0.868278 |

## Interpretation

The results are consistent with correct metric behavior:

1. The mean baseline has FVE exactly `0.0`, which is expected when the mean vector is computed from the same target tensor.
2. The zero baseline is strongly negative for the main model, meaning a zero vector is much worse than the mean activation baseline.
3. The shuffled baseline has FVE close to `-1`, meaning another example's activation is substantially worse than the mean vector but still far better than a zero vector for the main-model artifacts.
4. The context-1024 main-model artifact has lower mean-baseline MSE than context-512, suggesting that no-truncation activations are more tightly clustered in this pilot set. This may be useful later, but `ctx512` remains an efficient near-no-truncation choice.
5. The small model has a much larger mean-baseline MSE than the main model pilot artifacts, so activation scale/variance differs substantially across model sizes.

## Git Artifact Policy

Large and generated artifacts are not committed:

- `outputs/` is ignored in `.gitignore`.
- Tensor files such as `.pt`, `.npy`, `.npz`, and `.safetensors` are ignored.
- Raw and processed datasets are ignored.

This is the correct default. The repository should track source code, configs, experiment logs, and compact documentation summaries. If later a small final metrics table or figure is needed for review, it should be copied into `docs/` or a non-ignored report directory intentionally.

## Decision

Proceed to **Phase 5: Activation Reconstructor (AR)**.

Phase 5 should train a first text-to-activation reconstructor using available text fields, starting with the pilot artifacts and then scaling once the training/evaluation loop is validated.
