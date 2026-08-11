# Input package

This directory contains reproducible **configuration inputs**, not redistributed ZTF photometry.

## Files
- `config/experiment.yaml`: manuscript-aligned 60,000-object maximum-target configuration.
- `config/class_mapping.json`: ALeRCE subclass-to-broad-class mapping.
- `manifests/object_manifest_template.csv`: schema for the executed object manifest.

The experiment retrieves public ZTF detections through ALeRCE. The exact executed cohort must be archived as an object-ID manifest after retrieval. Protocol targets must not be reported as executed counts.
