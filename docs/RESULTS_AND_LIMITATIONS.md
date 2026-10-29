# Executed results and interpretation

- Prepared dataset: 18,000 valid ALeRCE-pseudolabelled objects, 6,000 per broad class. Train=11,700; validation=2,699; test=3,601, object-disjoint protocol.
- Frozen manifest SHA-256: `17880650d5f02b7d3b03cf76c20bf769bbe635156f2c02a864dafdb759080828`.
- Six full-data classical models (mean held-out macro-F1): RF 0.8340; BiGRU 0.8240; CNN 0.8132; Transformer 0.7908; Classical24b 0.7907; Classical24a 0.7844.
- Matched 3,000-object training budget (mean held-out macro-F1): Quantum24 0.7625; Classical24a 0.7659; Classical24b 0.7710. No demonstrated quantum advantage.
- Corrected TLC model: 423,220 generated, 69,765 distinct states, configured checks passed. `-deadlock` flag was used (deadlock checking disabled).
- Targeted TLC mutants: gate bypass (45 distinct), stale assessment (689), retry assessment (45), cross-object isolation (5). These intentionally faulty versions generated expected counterexamples.
- Local test run: 4 pytest tests passed in 1.89 seconds.

## Methodology and limits

All main labels originate from the ALeRCE broker and are *not* independently confirmed. The balanced class-filtered, detection-filtered selection cannot support claims about natural class prevalence or alert-stream accuracy. Prefix tests reuse objects selected for having at least 20 observations. The quantum-classical comparison uses fewer training examples than the full classical comparison, and that distinction must be explicit in every table and figure.

The code includes an independent-label evaluation entry point, but no verified external-label subset has been supplied; do not report independent astronomical ground-truth accuracy. TLC checked a finite workflow abstraction, not the numerical correctness of the classification network. Model-checking outcomes are limited to the specified configurations and abstraction.

Locally retained compact results ZIPs include held-out prediction files, confusion counts, uncertainty summaries, paired comparisons and model metadata. Confirm their presence on the Zenodo record before claiming that each file is publicly downloadable. Do not infer a performance claim from this summary alone; refer to primary CSV/JSON evidence. The full photometry and checkpoint dataset is not included in this Git repository.

## Measurement boundaries

The quality-weight sensitivity sweep was executed on fixed held-out predictions with confidence, observational quality and source-consistency checks; it does not establish the effect of adding ensemble uncertainty to that combined gate. Inference-time columns are **offline per-object measurements**, not end-to-end alert-broker latency or deployed-system throughput. The earlier 60,000-object results and 9,000-object test set belong to a historical, differently specified analysis and are not results from the 18K run.
