# Validation status and boundaries

The code repository includes source programs, experiment configurations, software tests, TLA+ specifications and reproduction instructions.

**Software:** The packaged source passed `bash -n mac_run.sh` and four Python tests (`tests/test_18k.py`, `tests/test_mac_fresh.py`). These tests used synthetic fixtures and fake broker replies, not real astronomical observations.

**Separate measured Mac experiment (29 October 2026):** The user-reported run processed 18,000 broker-pseudolabelled objects; the object-disjoint split was 11,700/2,699/3,601. Six classical models ran for five seeds each with 11,700 training objects. A matched 3,000-object study ran Quantum24 and two Classical24 controls for five seeds. Locally retained report ZIPs include predictions, confusion matrices, paired tests, quality-score sweeps, and TLC logs.

**Formal execution:** TLC on the corrected workflow generated 423,220 states and found 69,765 distinct states, with no configured property violation. Four faulty workflow variants triggered their expected properties. The TLC command used `-deadlock`, so deadlock freedom was **not** checked. Neither TLC nor software unit tests prove astronomical classifier correctness.

**Missing or unmeasured:** no independent confirmed-label benchmark, naturally imbalanced alert-stream evaluation, live broker integration or production end-to-end latency. Retrospective short prefixes are not an independent low-detection cohort.

**Repository limits:** No full light curves, checkpoints, held-out prediction CSVs or TLC logs are committed to GitHub. The depositor supplied Zenodo DOI https://doi.org/10.5281/zenodo.23770646, but the record's individual files and access rights have not been independently checked. A historical `train_ERROR.log` documents an earlier resolved PennyLane installation problem, not the final training status.
