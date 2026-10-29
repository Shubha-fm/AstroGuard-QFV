# Research archive and evidence files

- **Depositor-provided Zenodo DOI:** [10.5281/zenodo.23770646](https://doi.org/10.5281/zenodo.23770646).
- **Source code:** [Shubha-fm/AstroGuard-QFV](https://github.com/Shubha-fm/AstroGuard-QFV).
- **Availability:** inspect the Zenodo record's actual file list and license before saying that any individual archive or dataset is downloadable. The record contents and rights were not independently confirmed at the time of this edit.

This directory keeps SHA-256 hashes for the three locally retained compact evidence ZIPs in [`SHA256SUMS.txt`](SHA256SUMS.txt). Their names are listed in the repository README. A newly created cleaned `.tar.gz` archive has a different checksum and must be checked against its own SHA-256 value.

Do not store the approximately 3 GB working dataset, model checkpoints, local SQLite collection state, or manuscript files in ordinary Git history. The 18K class labels are ALeRCE broker-generated pseudolabels; they are not independent astronomical ground truth.
