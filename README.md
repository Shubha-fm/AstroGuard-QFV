# AstroGuard-QFV

**Hybrid Quantum–Transformer Classification of ZTF Light Curves with Formally Verified Release Semantics**

This repository contains the software, input configuration, output structure, and TLA+ verification artifacts associated with the *Astronomy and Computing* manuscript:

> **AstroGuard-QFV: Hybrid Quantum–Transformer Classification of ZTF Light Curves with Formally Verified Release Semantics**

## Overview

AstroGuard-QFV is an assurance-oriented framework for classifying Zwicky Transient Facility (ZTF) light curves. It combines:

- a Transformer for irregular astronomical time-series representation;
- a residual four-qubit variational quantum feature map;
- a parameter-matched classical bottleneck for controlled comparison;
- a five-member deep ensemble for uncertainty estimation;
- selective prediction using confidence, uncertainty, data quality, and source-validity conditions; and
- TLA+ specifications for verifying release semantics under concurrency, retries, model-version changes, and stale assessments.

The framework separates **predictive evidence** from **workflow assurance**. TLA+ verifies the release-control logic; it does not prove the correctness of the neural or quantum classifier.

## Repository Structure

```text
AstroGuard-QFV/
├── programs/
│   ├── astroguard_qfv.py
│   ├── astroguard_qfv_fast_12000.py
│   ├── generate_results_macros.py
│   ├── run_experiment.sh
│   ├── run_tlc.sh
│   ├── AstroGuardQFV.tla
│   ├── AstroGuardQFV.cfg
│   └── PROGRAMS.txt
│
├── input/
│   ├── config/
│   │   ├── experiment.yaml
│   │   └── class_mapping.json
│   ├── manifests/
│   │   └── object_manifest_template.csv
│   └── README/
│       └── README_INPUT.md
│
└── output/
    ├── results/
    │   ├── manuscript_values.json
    │   ├── manuscript_values.csv
    │   ├── final_summary.template.json
    │   └── predictions_schema.csv
    ├── plots/
    ├── models/
    ├── tlc/
    └── README_OUTPUT.md
```

## Dataset Protocol

The large-scale protocol targets a maximum of **60,000 ZTF objects**, balanced across three broad classes:

| Broad class | Included subclasses | Maximum target |
|---|---|---:|
| Transient | SN, SNIa, SNIbc, SNII | 20,000 |
| Periodic | Periodic-Other, RR Lyrae, EB, LPV | 20,000 |
| Stochastic | AGN, QSO, Blazar, YSO | 20,000 |

If the maximum cohort is fully attained, the planned object-disjoint split is:

- Training: 42,000 objects
- Validation: 9,000 objects
- Test: 9,000 objects

These are **protocol targets**, not automatically executed counts. The exact cohort used in a study must be recorded in the final object manifest.

## Main Model Configuration

- Maximum sequence length: 128 observations
- Input channels: 6
- Transformer layers: 4
- Model dimension: 128
- Attention heads: 4
- Feed-forward dimension: 256
- Dropout: 0.15
- Quantum circuit: 4 qubits
- Variational layers: 2
- Fused feature dimension: 132
- Ensemble members: 5
- Ensemble seeds: 42, 142, 242, 342, 442

The multi-objective training loss is:

```text
0.10 × masked reconstruction
+ 0.10 × contrastive learning
+ 0.80 × supervised classification
```

## Initial Release-Gate Configuration

The initial validation configuration is:

```text
confidence threshold       = 0.50
epistemic uncertainty max  = 0.10
quality threshold          = 0.30
```

Final thresholds should be selected using validation data only.

## Installation

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install alerce numpy pandas torch scikit-learn pennylane \
    pennylane-lightning matplotlib tqdm PyYAML
```

TLA+ model checking additionally requires Java and the TLA+ tools JAR.

## Run the Main Experiment

From the repository root:

```bash
python programs/astroguard_qfv.py
```

A reduced-cost pilot is also provided:

```bash
python programs/astroguard_qfv_fast_12000.py
```

The fast version uses a smaller cohort for debugging and preliminary validation. Results from the pilot should not be reported as large-scale results unless the manuscript explicitly states that experimental scale.

## Run TLA+ / TLC

Set the path to `tla2tools.jar`, then run:

```bash
TLA2TOOLS_JAR=/path/to/tla2tools.jar bash programs/run_tlc.sh
```

The corrected specification checks release-control properties including:

- `TypeOK`
- `NoUnsafeRelease`
- `NoStaleRelease`
- `RetryInvalidatesAssessment`
- `ObjectIsolation`
- `AssessedProgress`

The formal evaluation should also include fault-injected variants for:

1. gate bypass;
2. stale release after model reload;
3. retry-state retention; and
4. cross-object state contamination.

Raw TLC logs should be saved under:

```text
output/tlc/
```

## Input Files

The `input/` folder contains configuration files and manifest schemas.

The repository does **not** redistribute ZTF photometric data. Light curves are retrieved through public ALeRCE services.

For reproducibility, archive the exact executed object manifest containing:

- ZTF object ID;
- original ALeRCE class;
- broad class;
- retrieval timestamp;
- number of valid detections;
- split assignment;
- ALeRCE client/classifier version; and
- preprocessing configuration hash.

## Output Files

The `output/` folder contains the expected output structure.

Important files include:

```text
output/results/manuscript_values.json
output/results/manuscript_values.csv
output/results/final_summary.template.json
output/results/predictions_schema.csv
```

`manuscript_values.json` contains protocol and configuration values used in the manuscript.

Measured quantities such as:

- accuracy;
- balanced accuracy;
- macro-precision;
- macro-recall;
- macro-F1;
- AUROC;
- ECE;
- Brier score;
- coverage;
- released-set accuracy;
- selective risk;
- TLC state counts; and
- TLC runtime

must come from executed experiment and verification artifacts.

## Expected Executed Outputs

A complete journal experiment should generate:

```text
output/results/final_summary.json
output/results/ensemble_predictions.csv
output/results/dataset_summary.csv
output/results/reliability_bins.csv
output/results/risk_coverage.csv

output/plots/confusion_matrix.png
output/plots/reliability_diagram.png
output/plots/risk_coverage.png
output/plots/confidence_uncertainty.png

output/models/
output/tlc/
```


## Manuscript Repository

GitHub:

https://github.com/Shubha-fm/AstroGuard-QFV

## Authors

**Shubha Chakraborty**  
Department of Computer Science  
The University of Burdwan, West Bengal, India

**Rahul Karmakar**  
Department of Computer Science  
The University of Burdwan, West Bengal, India



