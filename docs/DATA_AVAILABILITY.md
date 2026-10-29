# Data availability and external archive

This GitHub repository deliberately excludes the ~3 GB raw light-curve download, the collection SQLite checkpoint, `full_sequences.npy`, and trained-model checkpoint tensors.

The complete-data collection is a balanced 18,000-object ALeRCE/ZTF cohort. All reported broad classes are ALeRCE broker-generated labels, not independently confirmed ground truth. The separately retained collection ZIP contains metadata and a freeze statement, **not** the 18,000 raw light curves.

The project depositor has supplied the Zenodo research-archive identifier **[10.5281/zenodo.23770646](https://doi.org/10.5281/zenodo.23770646)**. This is a version-specific archive identifier; consult the record page for its current file list, sizes, checksums, access conditions, version, and stated license. We could not independently confirm those fields when this repository link was updated, so the DOI alone is **not** evidence that every raw light curve, checkpoint, or report ZIP is publicly downloadable.

Before reusing or redistributing the downloaded light curves and broker classifications, check the applicable source data terms and any additional author permissions. The existence of a public DOI does not by itself resolve redistribution rights. The depositor should confirm that the uploaded archive excludes unnecessary collection-progress databases, local machine metadata, and historical training error logs, or explain any retained files. Use a checksum for the exact deposited tarball; do not reuse the hashes of other report ZIP versions.

The compact analysis packages can be reproduced from the source code with access to the same frozen input records and random seeds. Redownloading from ALeRCE may not reproduce the exact dataset if the broker changes classifications or service output.
