# Phase 2 Results: Dataset Preparation

## Status

**Accepted / successful.**

The dataset preparation pipeline successfully converted local raw datasets into project-standard processed JSONL files and produced a dataset manifest.

## Command Run

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

## Generated Files

```text
data/processed/pilot_100.jsonl
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test_indomain.jsonl
data/processed/test_surface_shift.jsonl
data/processed/test_language_shift.jsonl
data/processed/dataset_manifest.json
```

## Output Counts

| File | Rows |
|---|---:|
| `pilot_100.jsonl` | 100 |
| `train.jsonl` | 5000 |
| `validation.jsonl` | 500 |
| `test_indomain.jsonl` | 500 |
| `test_surface_shift.jsonl` | 500 |
| `test_language_shift.jsonl` | 361 |
| **Total JSONL rows** | **6961** |

## Manifest Summary

Manifest file: `data/processed/dataset_manifest.json`

```json
{
  "schema_version": "phase2_v1",
  "generated_at": "2026-06-05T14:14:43.430831+00:00",
  "output_counts": {
    "pilot_100.jsonl": 100,
    "train.jsonl": 5000,
    "validation.jsonl": 500,
    "test_indomain.jsonl": 500,
    "test_surface_shift.jsonl": 500,
    "test_language_shift.jsonl": 361
  },
  "counts_by_language": {
    "cpp": 164,
    "java": 164,
    "python": 6633
  },
  "counts_by_transformation_type": {
    "language_shift": 361,
    "original": 6100,
    "rename_identifiers": 500
  }
}
```

## Warnings

The run produced expected dataset-size warnings for HumanEval-X:

- `humaneval_x_python:test requested 500 examples after filtering, but only 33 were available.`
- `humaneval_x_cpp:test requested 500 examples after filtering, but only 164 were available.`
- `humaneval_x_java:test requested 500 examples after filtering, but only 164 were available.`

These warnings are acceptable. HumanEval-X is a controlled multilingual evaluation dataset, so fewer examples are expected. The language-shift test set still contains 361 examples across Python, C++, and Java.

## Example Schema Check

A sample `train.jsonl` row contains the expected fields:

- `example_id`
- `source_dataset`
- `source_split`
- `split`
- `language`
- `task_family`
- `code`
- `prompt`
- `reference_description`
- `transformation_type`
- `paired_example_id`
- `metadata`

The prompt follows the project template:

```text
Read the following function and prepare to explain what it does.

<code>
...
</code>

Explanation:
```

A sample `test_language_shift.jsonl` row also includes a HumanEval-X task id in metadata and uses `transformation_type = language_shift`.

## Code Review Notes

The implemented code provides:

- Local dataset loading from `datasets.load_from_disk`.
- Project-standard JSONL writing.
- Deterministic shuffling via seed.
- Conservative code normalization.
- Conservative Python identifier renaming for surface-shift examples.
- Manifest generation with counts and warnings.
- Clear sectioned logs.

The Phase 1 `torch_dtype` deprecation warning was also addressed by adding a `load_causal_lm_with_dtype` helper that tries `dtype=` first and falls back to `torch_dtype=` if needed.

## Decision

Proceed to **Phase 3: Activation Extraction**.

The next phase should extract selected hidden-state activations for at least the pilot dataset first, then scale to train/validation/test splits once the output format and memory usage are validated.
