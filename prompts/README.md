# Implementation Prompts

This directory will store prompts used with Codex or other coding agents.

Each prompt should be saved as a Markdown file with:

- Goal
- Target files
- Required behavior
- Constraints
- Test instructions
- Expected output
- Official NLA reference alignment

Before implementing AV, AR, activation injection, or reconstruction metrics, prompts must instruct the coding agent to read:

- `docs/source_usage.md`
- `docs/reference_mapping.md`
- Official NLA repository: https://github.com/kitft/natural_language_autoencoders

The coding agent should use the official repository as a reference implementation, not as a default source for copied code. Any copied or closely adapted code must include attribution in the target file.

Planned prompt sequence:

1. `01_feasibility_probe.md`
2. `02_dataset_preparation.md`
3. `03_activation_extraction.md`
4. `04_baselines_and_metrics.md`
5. `05_reference_aligned_ar_design.md`
6. `06_reference_aligned_av_design.md`
7. `07_full_nla_loop.md`
8. `08_evaluation_and_report.md`
