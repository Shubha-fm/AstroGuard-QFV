# AstroGuard-QFV

**Trustworthy Quantum–Classical Learning for Astronomical Time-Series Classification with Formal Release Assurance**

This repository provides supporting implementation, configuration, output-schema, and formal-model resources associated with the *Astronomy and Computing* manuscript:

> **AstroGuard-QFV: Trustworthy Quantum–Classical Learning for Astronomical Time-Series Classification with Formal Release Assurance**

## Overview

AstroGuard-QFV is a trustworthy astronomical time-series classification framework for Zwicky Transient Facility (ZTF) light curves. It combines:

- a Transformer for irregular astronomical time-series representation;
- a residual four-qubit variational quantum feature branch;
- a parameter-matched classical residual control;
- a five-member deep ensemble for uncertainty estimation;
- selective prediction using confidence, uncertainty, observational quality, and source-consistency conditions; and
- TLA+ release-workflow modeling for concurrency, retries, model-version changes, stale assessments, and cross-object interference.

The framework deliberately separates **predictive evidence** from **workflow assurance**. TLA+ constrains the stateful release-control logic; it does not prove the correctness of the learned classifier.

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
│   ├── manifests/
│   └── README/
│
└── output/
    ├── results/
    ├── plots/
    ├── models/
    ├── tlc/
    └── README_OUTPUT.md
```

## Dataset Protocol

The large-scale protocol targets a maximum of **60,000 ZTF objects**, balanced across three broad classes.

| Broad class | Included subclasses | Maximum target |
|---|---|---:|
| Transient | SN, SNIa, SNIbc, SNII | 20,000 |
| Periodic | Periodic-Other, RR Lyrae, EB, LPV | 20,000 |
| Stochastic | AGN, QSO, Blazar, YSO | 20,000 |

When the full cohort is attained, the object-disjoint split is:

- Training: 42,000 objects
- Validation: 9,000 objects
- Test: 9,000 objects

Exact executed object identities and split assignments should be preserved in a fixed manifest because ALeRCE is a live service.

## Main Model Configuration

- Maximum sequence length: 128 observations
- Input channels: 6
- Transformer layers: 4
- Model dimension: 128
- Attention heads: 4
- Feed-forward dimension: 256
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

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install alerce numpy pandas torch scikit-learn pennylane \
    pennylane-lightning matplotlib tqdm PyYAML
```

TLA+ model checking additionally requires Java and `tla2tools.jar`.

## Run the Main Experiment

```bash
python programs/astroguard_qfv.py
```

A reduced-cost implementation is also available for debugging and pilot runs:

```bash
python programs/astroguard_qfv_fast_12000.py
```

Pilot outputs must not be presented as the large-scale evaluation unless the manuscript explicitly uses that scale.

## TLA+ / TLC Model

The corrected specification is in:

```text
programs/AstroGuardQFV.tla
programs/AstroGuardQFV.cfg
```

The final formal model checks:

- `TypeOK`
- `NoUnsafeRelease`
- `FreshRelease`
- `RetryInvalidatesAssessment`
- `ObjectIsolation`

`FreshRelease`, `RetryInvalidatesAssessment`, and `ObjectIsolation` are transition/action properties. The model includes explicit release, review, withholding, retry, model-reload, and concurrent object-local actions.

The manuscript also evaluates deliberately faulty variants representing:

1. gate bypass;
2. stale release after a model reload;
3. retry-state retention; and
4. cross-object state contamination.

Only executed TLC outputs should be placed under `output/tlc/` and reported as measured verification results.

## Reproducibility Policy

Files under `output/results/` distinguish configuration/protocol values from executed measurements.

**Do not treat protocol/configuration values as measured results.**

Measured quantities such as predictive metrics, calibration values, selective-prediction statistics, TLC state counts, and counterexample lengths should be populated only from executed experiment or verification artifacts.

## Data Availability

ZTF detections are accessed through public ALeRCE services. This repository does not redistribute the underlying ZTF photometric data.

For exact reproducibility, preserve the executed object manifest containing at least:

- ZTF object ID;
- original ALeRCE class;
- broad class;
- retrieval timestamp;
- valid-detection count;
- split assignment;
- ALeRCE client/classifier version; and
- preprocessing configuration hash.

## Authors

**Shubha Chakraborty**  
Department of Computer Science  
The University of Burdwan, West Bengal, India

**Rahul Karmakar**  
Department of Computer Science  
The University of Burdwan, West Bengal, India
