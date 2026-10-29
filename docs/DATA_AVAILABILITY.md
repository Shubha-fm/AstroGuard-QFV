# Data availability and external archive

This GitHub repository deliberately excludes the ~3 GB raw light-curve download, the collection SQLite checkpoint, `full_sequences.npy`, and trained-model checkpoint tensors.

The complete-data collection is a balanced 18,000-object ALeRCE/ZTF cohort. All reported broad classes are ALeRCE broker-generated labels, not independently confirmed ground truth. The separately retained collection ZIP contains metadata and a freeze statement, **not** the 18,000 raw light curves.

The author can separately archive the large dataset and trained checkpoints with a DOI, subject to the providers' usage and redistribution terms. Record a checksum and version the archive. Until that archive exists, do not claim that the raw data are publicly downloadable.

The compact analysis packages can be reproduced from the source code with access to the same frozen input records and random seeds. Redownloading from ALeRCE may not reproduce the exact dataset if the broker changes classifications or service output.
