# Reviewer evidence status — AstroGuard-QFV 18K (29 October 2026)

The following status applies to the completed 18,000-object experiment. It replaces an earlier plan that labelled completed runs as pending. Independently check the Zenodo record inventory before claiming that every local file is public.

| Reviewer question | Recorded 18K evidence | Outstanding limitation |
| --- | --- | --- |
| Dataset composition | Completed: 6,000 objects in each of three broad classes; 11,700/2,699/3,601 object-disjoint split | Not a naturally imbalanced, unranked ZTF alert stream |
| Label provenance | Broker-generated ALeRCE pseudolabels used and documented | No independent spectroscopic/confirmed labels in the main evaluation |
| Classical baselines | Completed: RF, CNN, BiGRU, Transformer and two Classical24 models, five seeds each, 11,700 training objects | Avoid comparison as equal-budget models with the 3,000-object matched study |
| Matched quantum controls | Completed: Quantum24, Classical24a, Classical24b with equal 3,000-object training budgets and five seeds | No demonstrated quantum accuracy or hardware advantage |
| Confusion and significance | Completed: test predictions, confusion counts and object-level paired comparisons | Make comparisons using the same held-out objects; distinguish seed variation from object uncertainty |
| Quality score | Completed: quality-weight sensitivity on held-out predictions | Sweep uses confidence, quality and source checks; not a full ensemble-disagreement gate |
| Early classification | Completed: retrospective 2, 5, 10, 20, 40, 80, 128 observation prefixes | Truly sparse new alerts unmeasured; cohort was selected with >=20 eventual detections |
| Model cost | Offline per-object inference times recorded | End-to-end broker latency, throughput and live deployment unmeasured |
| Formal workflow | TLC corrected case PASS, 69,765 distinct states; four targeted mutations detected | Finite abstraction, not a classifier correctness proof; deadlock checking disabled by `-deadlock` |
| Manuscript and response | Prior manuscript draft and reviewer response exist | Must replace old 60,000-object/9,000-test historical results and pending-work text with actual 18K evidence |

**Scientific limits are not defects to hide.** State them explicitly. The Zenodo record DOI supplied by the author is https://doi.org/10.5281/zenodo.23770646.
