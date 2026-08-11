# Output files

This package now contains all **protocol and configuration values used in the manuscript**.

## Included
- `results/manuscript_values.json`
- `results/manuscript_values.csv`
- `results/final_summary.template.json`
- `results/predictions_schema.csv`

The protocol values include:
- 60,000 maximum target objects
- 20,000 objects per broad class
- 42,000 / 9,000 / 9,000 target split if the full cohort is attained
- sequence length 128
- 6 input channels
- 4-layer Transformer, d=128, 4 heads, FFN=256
- 4-qubit, 2-layer VQC
- 5-member ensemble and seeds
- optimizer/training settings
- release thresholds
- quality-score weights
- formal properties and seeded fault models

## Important
Accuracy, macro-F1, AUROC, calibration, coverage, TLC state counts, and TLC runtime are marked `NOT_EXECUTED` because no executed experiment summary or TLC log is currently available. They must be replaced only after the real run.
