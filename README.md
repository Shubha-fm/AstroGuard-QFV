# AstroGuard-QFV: 18K astronomical time-series experiment

Reproducibility code and compact evidence for **AstroGuard-QFV**, a quantum-classical and formal-assurance study of astronomical light-curve classification.

This repository contains the **fresh macOS 18,000-object experiment**. It replaces earlier prototype scripts and placeholder output tables. The 3 GB light-curve dataset, memory-mapped arrays, raw checkpoints, and optional model weights are intentionally **not stored in ordinary Git history**.

## Experiment status (29 October 2026)

| Evidence | Outcome |
| --- | --- |
| Data collection | 18,000 objects; 6,000 Periodic, 6,000 Stochastic, 6,000 Transient |
| Dataset preprocessing | 18,000 valid; 0 excluded at preparation |
| Object-disjoint split | Train 11,700; validation 2,699; test 3,601 |
| Classical baselines | 6 models × 5 seeds, trained and analyzed |
| Matched comparison | Quantum24, Classical24a and Classical24b × 5 seeds |
| Software tests | 4 passed (macOS run, 29 October 2026) |
| TLC model checking | 69,765 distinct corrected-model states, configured checks passed |
| TLC mutation tests | Four faulty workflows detected as expected |

**Important scientific limitations.** All main-cohort labels are ALeRCE *broker-generated pseudolabels*; they are not independent astronomical ground truth. Candidate sampling is class-filtered and the 20-detection inclusion rule does not reflect the natural alert stream. The matched quantum model **did not show an accuracy advantage** over the 24-parameter classical alternatives. TLC results apply to a finite release-workflow model; they do not certify an astronomical classifier or establish deployed-system correctness. The TLC runs use `-deadlock`, which disables deadlock checking, so deadlock freedom is not claimed.

## Repository layout

- `mac_run.sh`: macOS commands for setup, collection, training, analysis, model checking, and tests.
- `run_18k_collect.py`, `run_18k.py`, `run_large.py`, `run_18k_external_labels.py`: experiment entry points.
- `suite/`: data preparation, classical/quantum models, experiments, metrics, and TLC wrapper.
- `tla/`: corrected workflow, faulty variants, and TLC configurations.
- `configs/`: experimental protocol configuration.
- `tests/`: offline tests and fixtures.
- `input/`: empty templates and format documentation (not real photometry).
- `artifacts/`: report checksums and instructions; large report archives are archived separately.
- `docs/`: execution details, statistical interpretation, and data limitations.

## Reproduce the pipeline on a Mac

Use Python 3.11 or 3.12. Clone the repository and run:

```bash
bash mac_run.sh setup
bash mac_run.sh check
bash mac_run.sh query 100 40
bash mac_run.sh collect 1000
# Repeat collection until READY_FOR_PREPARE; expand candidate pages if needed
bash mac_run.sh prepare
bash mac_run.sh smoke
bash mac_run.sh classical
bash mac_run.sh analyze-classical
bash mac_run.sh setup-quantum
bash mac_run.sh matched
bash mac_run.sh analyze-matched
bash mac_run.sh tests
```

For formal verification, download the official **tla2tools.jar** from the [TLA+ releases](https://github.com/tlaplus/tlaplus/releases) into this directory, install Java, then run:

```bash
bash mac_run.sh verify
```

Running `query` or `collect` accesses the ALeRCE service and downloads real light-curve data. It is **not** required merely to inspect the compact evidence packages. Check `README_MAC_FRESH.md` for full setup instructions.

## Evaluated models and key findings

**Full-data classical study** (11,700 train / 2,699 validation / 3,601 test): mean test macro-F1 (five seeds) was RF **0.8340**, BiGRU **0.8240**, CNN **0.8132**, Transformer **0.7908**, Classical24b **0.7907**, and Classical24a **0.7844**. Descriptive means are not statistical evidence of superiority by themselves.

**Matched study** (3,000 train per model; the same validation/test cohort): mean test macro-F1 was Classical24b **0.7710**, Classical24a **0.7659**, and Quantum24 **0.7625**. Do not compare the matched-study results to 11,700-object classical baselines as if they used the same training budget.

The exact per-seed metrics, held-out predictions, early-prefix curves, quality sensitivity and verification logs are in the archived evidence files below. Report the actual 3,601-object test size, not a rounded 3,600.

## Evidence files and provenance

The results archives have **not yet been uploaded to this repository**.
The publication-ready archives are:
- AstroGuard_QFV_18K_Collection_Summary.zip
- AstroGuard_QFV_18K_Classical_Results.zip
- AstroGuard_QFV_18K_Matched_Results.zip

See `artifacts/SHA256SUMS.txt` to verify locally held copies.
Raw photometry (~3 GB) is excluded from Git and awaits a permitted research-data deposit.
This repository reports data-provider pseudolabel agreement, not independent ground truth.

## Reuse and citation

Source authorship and data-provider attribution should be retained in any derived work. No license is asserted by this cleanup; the authors should select and add an explicit code license before claiming unrestricted reuse. This repository is associated with a manuscript under revision, not an accepted or published article. Cite the data sources and this repository version when reusing experimental outputs.
