# Data Directory

This directory documents the expected local data layout. Large datasets should not be committed to Git.

## Layout

```text
data/
├── raw/        # Downloaded source datasets, not committed
├── interim/    # Temporary intermediate files, not committed
└── processed/  # Processed train/validation/test files, not committed
```

## Planned Processed Schema

Each processed example should eventually contain:

- `example_id`
- `source_dataset`
- `split`
- `language`
- `task_family`
- `code`
- `prompt`
- `reference_description`
- `transformation_type`
- `paired_example_id`

## First Dataset Task

The first dataset task is to create a small pilot set of Python function-level prompts for activation extraction.

Initial target size:

- 100 examples for smoke testing
- 500 examples for the first extraction run
- 5,000 examples for the first training run, if the pipeline is stable
