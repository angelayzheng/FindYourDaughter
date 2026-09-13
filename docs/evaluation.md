# Evaluate detectors against the draft reference set

Run all registered algorithms with their default settings:

```powershell
python scripts/evaluate_detectors.py
```

This reads `eval_set/case_*/` and writes `nifti_previews/evaluation/report.json`,
`summary.md`, and separate prediction/diagnostic JSON files under `baseline/`,
`contact/`, and `fusion/`. The runner performs no training or parameter tuning;
no new dependencies or network access are involved. The reference files are read only, and output inside the
reference dataset is rejected. The required `run.py` command is unchanged.

Select an algorithm, case subset, or matching tolerance:

```powershell
python scripts/evaluate_detectors.py --dataset eval_set --detector contact --cases 19 23 --output-dir nifti_previews/evaluation_subset
python scripts/evaluate_detectors.py --dataset eval_set --tolerance-mm 4 --output-dir nifti_previews/evaluation_4mm
```

Use separate output directories for comparisons. Rerunning in the same folder
replaces the report and successful case exports. Files from earlier runs for
unselected or failed cases may remain; `report.json` identifies the cases and
detector runs actually included in the current score.

## Reference scope

The supplied set contains cases 19-23, with **19 draft daughter instances** on
1.5 mm native grids. Its own documentation says annotations need expert review
and may omit eligible origins. Results measure agreement with this draft set,
not official challenge performance or expert-validated accuracy. Unmatched
predictions need review; they are not automatically confirmed false positives.

Each case supplies `origN.nii.gz`, the **parent-only** `aortaN.nii.gz`,
`annotations.json`, and `daughtersN_draft.nii.gz`. Only CT and the parent mask
are passed to a detector. Daughter labels, guide points, manually selected
thresholds, confidence labels, and excluded candidates never tune or feed it.
The combined viewing mask is not a detector input.

The reader checks image/mask/label geometry, binary parent values, daughter-label
identity and nonoverlap with the parent, finite physical landmarks, metadata,
and voxel-to-LPS guide consistency when redundant guides are supplied. A full
run also checks case-directory inventory against the supplied manifest. Input,
backend, and scoring-module SHA-256 hashes and dependency versions are recorded
so results can be tied to the evaluated files. These checks do not validate anatomy.

All retained annotations participate regardless of qualitative confidence.
Review status and confidence accompany matched records. `excluded_candidate.json`
entries are withheld candidates, so they are not inserted as positive references
or treated as established negatives. Review-note locations are recorded per case.

## Measurements

- Match ostia **one to one** in physical SimpleITK LPS millimetres, maximizing
  the number of pairs within the tolerance before minimizing total distance.
  The default **3 mm** is a development tolerance, not an official scoring rule.
  IDs are local: matching `branch_001` to `branch_001` by name would be incorrect.
- Report matches, unmatched reference IDs, and unmatched prediction IDs with
  nearest-reference distances. `reference_precision = matches / predictions`,
  `reference_recall = matches / references`, and `reference_f1` are agreement
  statistics under this draft inventory. Overall counts are micro-averaged.
- For matched pairs only, report ostium and seed distance errors, angular
  direction error, and absolute radius error, with count/mean/median/maximum.
  Missing or search-envelope-limited radius estimates are excluded. The manual
  origin-diameter estimate is not substituted for the radius at the seed.
- Check whether the predicted seed's nearest native voxel has the matched
  daughter label, and measure its distance to the nearest voxel center of that
  label. A seed can be inside a labeled voxel with nonzero voxel-center distance.
  This is not distance to the continuous vessel surface. No Dice/IoU is reported:
  these detectors output landmarks and short paths, not daughter segmentation masks.

Null measurements and undefined denominators remain JSON `null`, not zero or
NaN. Per-case failures are recorded and make the CLI exit with code 1. Metrics
exclude failed cases and explicitly report successful/failed counts; do not
compare incomplete runs as full-set scores. Invalid invocation or inventory
errors exit with code 2. Complete runs exit with code 0.

Timing separates loading/reference validation from each detector call. Calls
run serially on CPU; inference timing includes any first-use lazy imports but
excludes matching, hashing, and output writes. It is not a four-core affinity
benchmark or an official runtime measurement.

## Measured results

All three detectors, including the experimental [fusion detector](fusion_detection.md),
were evaluated on 2026-09-13 with default settings, the supplied five-case draft
set, and a 3 mm tolerance. Baseline and contact reproduced the initial results.
The command was:

```powershell
python scripts/evaluate_detectors.py --detector all --output-dir nifti_previews/evaluation_20260913
```

The local `nifti_previews/evaluation_20260913/report.json` records full metrics,
input and implementation hashes, and dependency versions. Generated reports and
predictions remain ignored by version control.

| Measurement | Baseline | Contact | Fusion |
| --- | ---: | ---: | ---: |
| Matched references | 5 / 19 | 10 / 19 | 11 / 19 |
| Predictions | 8 | 16 | 19 |
| Unmatched predictions | 3 | 6 | 8 |
| Unmatched references | 14 | 9 | 8 |
| Reference precision | 62.5% | 62.5% | 57.9% |
| Reference recall | 26.3% | 52.6% | 57.9% |
| Reference F1 | 37.0% | 57.1% | 57.9% |
| Mean ostium error, matched only (mm) | 1.27 | 0.96 | 1.15 |
| Mean seed error, matched only (mm) | 0.99 | 0.95 | 1.00 |
| Mean direction error, matched only (degrees) | 12.51 | 14.39 | 14.24 |
| Seeds in their matched label | 5 / 5 | 10 / 10 | 11 / 11 |

Only three of the 19 annotations have usable numeric seed-radius references;
each detector matches just one of those. Radius error is therefore available
for only one pair per detector (0.15 mm baseline, 0.31 mm contact, 0.29 mm fusion),
insufficient for a general radius-quality comparison. Matched sets differ between algorithms,
so the other mean errors also describe different subsets.

| Case | Draft references | Baseline matches / predictions | Contact matches / predictions | Fusion matches / predictions |
| --- | ---: | ---: | ---: | ---: |
| 19 | 3 | 0 / 0 | 1 / 1 | 1 / 1 |
| 20 | 4 | 0 / 0 | 1 / 1 | 1 / 1 |
| 21 | 3 | 2 / 2 | 1 / 6 | 2 / 7 |
| 22 | 6 | 3 / 5 | 6 / 7 | 6 / 9 |
| 23 | 3 | 0 / 1 | 1 / 1 | 1 / 1 |

Contact matches more draft branches overall but performs worse on case 21 and
produces more unmatched candidates. Fusion matches one more reference than
contact on case 21, but adds two unmatched predictions on case 22. Its draft
recall and F1 are higher than contact's, while its precision is lower. All three
detectors completed all five cases without failures.
Review unmatched candidates and missed references, particularly case 21, before
using this set to tune detection or claiming improved anatomical accuracy.

## Tests and implementation

`evaluation/landmarks.py` implements matching and metrics;
`evaluation/draft_set.py` adapts the draft package and runs the registered
detectors. `scripts/evaluate_detectors.py` is the command-line entrypoint.
The existing synthetic scorer remains available independently.

```powershell
python -m unittest discover -s tests -p test_evaluation.py -v
```

Tests use independent temporary fixtures, so the ignored evaluation dataset is
not required in CI. They cover assignment conflicts, duplicate proposals,
tolerance boundaries, empty cases, missing radii, reflected/anisotropic physical
grids, label lookup, invalid references, inventory omissions, detector failures,
input preservation, and the detector/reference boundary.
