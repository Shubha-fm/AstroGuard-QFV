# AstroGuard-QFV: fresh 18,000-object experiment for macOS

This is a **new 18K study**, not a resumption of the 240K Windows project.
It creates a new SQLite database on your Mac. The archive contains code and
empty data directories, not real astronomical observations or new model results.
Use Terminal on an Intel or Apple Silicon Mac.

## A. Set up macOS

Extract `AstroGuard_18K_macOS_Fresh_Start.zip` to Downloads.
Open Terminal and enter:

```bash
cd ~/Downloads/AstroGuard_18K_macOS_Fresh_Start
```

Install Python **3.11** if necessary. With Homebrew:

```bash
brew install python@3.11
```

Without Homebrew, use the official Python.org macOS Python 3.11 installer.
If developer tools are requested, run `xcode-select --install`.

Then run:

```bash
bash mac_run.sh setup
bash mac_run.sh check
```

If Python 3.11 is not on PATH but Python 3.12 is present:

```bash
PYTHON_BIN=python3.12 bash mac_run.sh setup
```

Setup creates `.venv` and installs the core libraries and ALeRCE client.
The quantum dependency is installed separately. To test the broker import:

```bash
.venv/bin/python -c 'from alerce.core import Alerce; print("ALeRCE ready")'
```

## B. Discover candidate objects from scratch

Do this **before** downloading photometry. It queries ALeRCE ZTF class-filtered
pages and creates `input/collection_progress.sqlite3`:

```bash
bash mac_run.sh query 100 40
```

Run the same command repeatedly until the output says `PAGES_READY`.
`100` is the maximum requests per batch; `40` is the number of pages allowed
for each subclass, at 500 candidate IDs per page. Previously queried pages
are skipped on the next run. The fresh data target is **6,000 eligible
objects in each broad class**, totaling 18,000.

Some individual subclasses may have no matching objects. Querying a capped
class-filtered candidate frame is **not a random sample of ZTF alerts**.

## C. Download and check eligible light curves

```bash
bash mac_run.sh collect 1000
```

Repeat until the program says `18K cohort: READY_FOR_PREPARE`.
It saves the successful observations in `input/lightcurves/` and updates the
SQLite checkpoint after each attempt. If the candidate frame is insufficient,
expand the maximum pages and then run collection against the larger frame:

```bash
bash mac_run.sh query 100 80
bash mac_run.sh collect 1000 80
```

`PAGES_READY` means the configured page cap has been reached, not that all
available ZTF objects were enumerated. ALeRCE provides classifier predictions
rather than independently confirmed astrophysical ground-truth labels.

**Pause/resume:** Press Control+C, allow the Terminal prompt to return, and
restart the same command later. Only one downloader should run against the
SQLite file. Never delete `input/` if you want to resume.

To back up after the collector is stopped:

```bash
.venv/bin/python - <<'PYTHON_SNIPPET'
import sqlite3
s = sqlite3.connect('input/collection_progress.sqlite3')
d = sqlite3.connect('input/collection_progress_backup.sqlite3')
s.backup(d)
d.close()
s.close()
print('SQLite backup saved')
PYTHON_SNIPPET
```

## D. Freeze the exact cohort and split the data

```bash
bash mac_run.sh prepare
```

When all 18,000 objects have passed checks, `input/objects_18000.csv` and its
SHA-256 are frozen. Expected split:

- Training: **11,700** objects (65%)
- Validation: **2,700** objects (15%)
- Testing: **3,600** objects (20%)

These are intended sizes, not completed measurements. Verify the actual
counts in `RESULTS_18K/prepared/dataset_report.json`.

## E. Classical baselines and statistical analysis

Start with a small software test (NOT a manuscript result):

```bash
bash mac_run.sh smoke
```

Then run six classical models for five seeds, followed by statistical analysis:

```bash
bash mac_run.sh classical
bash mac_run.sh analyze-classical
```

The suite includes Random Forest, CNN, BiGRU, Transformer, and two
24-parameter residual controls. All use the frozen object-disjoint cohort.

## F. Quantum/classical parameter-matched study

```bash
bash mac_run.sh setup-quantum
bash mac_run.sh quantum-sanity
bash mac_run.sh matched
bash mac_run.sh analyze-matched
```

`quantum-sanity` is a short software test, not a scientific result. The
`matched` experiment compares the quantum 24-rotation branch against two
24-weight classical controls using the same 3,000-object training budget and
the same held-out data. The quantum simulator uses CPU on macOS in this code.
Do not compare it against the 11,700-object classical models as though their
training sample sizes were equal.

## G. Formal verification and independent labels

For TLA+/TLC, install Java, download the official `tla2tools.jar`, and put it
beside `mac_run.sh`. Then run:

```bash
bash mac_run.sh verify
```

Record actual TLC logs. When testing the cross-object fault, verify that the
counterexample demonstrates the targeted isolation property.

A separate independently confirmed label evaluation is available via
`run_18k_external_labels.py`. It requires a real external catalog with
provenance; broker predictions do not count as independent confirmation.

## H. Files to send back

- `RESULTS_18K/AstroGuard_QFV_18K_Collection_Summary.zip`
- `RESULTS_18K/classical/AstroGuard_QFV_18K_Classical_Results.zip`
- `RESULTS_18K/matched/AstroGuard_QFV_18K_Matched_Results.zip`

## Scientific limitations

Class-filtered ZTF samples cannot establish natural alert prevalence.
Objects must have 20 valid detections to enter the main cohort. Evaluating
early prefixes from those objects does not establish accuracy for genuinely
new low-detection alerts. Quality-weight sweeps and statistical confidence
intervals must be computed using actual held-out predictions; do not invent
results. Separate independently validated labels and a natural-prevalence
alert cohort are still necessary to fully address reviewer concerns.
