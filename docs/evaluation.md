# Evaluate detectors against the draft reference set

Run all registered algorithms with their default settings:

```powershell
python scripts/evaluate_detectors.py
```

This reads `eval_set/case_*/` and writes `nifti_previews/evaluation/report.json`,
`summary.md`, and separate prediction/diagnostic JSON files under `baseline/`,
`contact/`, `fusion/`, and `refined/`. The runner performs no training or parameter tuning;
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

All four detectors, including the experimental [refined detector](refined_detection.md),
were evaluated on 2026-09-13 with default settings, the supplied five-case draft
set, and a 3 mm tolerance. Baseline, contact, and fusion predictions are identical
to their saved results before refinement; their implementations and the default
selection remain unchanged. `refined` is a separate opt-in algorithm.
The command was:

```powershell
python scripts/evaluate_detectors.py --detector all --output-dir nifti_previews/improvement_refined
```

The local `nifti_previews/improvement_refined/report.json` records full metrics,
input and implementation hashes, and dependency versions. Generated reports and
predictions remain ignored by version control.

| Measurement | Baseline | Contact | Fusion | Refined |
| --- | ---: | ---: | ---: | ---: |
| Matched references | 5 / 19 | 10 / 19 | 11 / 19 | 13 / 19 |
| Predictions | 8 | 16 | 19 | 17 |
| Unmatched predictions | 3 | 6 | 8 | 4 |
| Unmatched references | 14 | 9 | 8 | 6 |
| Reference precision | 62.5% | 62.5% | 57.9% | 76.5% |
| Reference recall | 26.3% | 52.6% | 57.9% | 68.4% |
| Reference F1 | 37.0% | 57.1% | 57.9% | 72.2% |
| Mean ostium error, matched only (mm) | 1.27 | 0.96 | 1.15 | 1.15 |
| Mean seed error, matched only (mm) | 0.99 | 0.95 | 1.00 | 0.99 |
| Mean direction error, matched only (degrees) | 12.51 | 14.39 | 14.24 | 14.39 |
| Seeds in their matched label | 5 / 5 | 10 / 10 | 11 / 11 | 13 / 13 |

Only three of the 19 annotations have usable numeric seed-radius references;
baseline, contact, and fusion each match just one of those (radius errors of
0.15, 0.31, and 0.29 mm, respectively). Refined matches two, with a mean absolute
radius error of 0.21 mm. These samples are insufficient for a general
radius-quality comparison. Matched sets differ between algorithms,
so the other mean errors also describe different subsets.

| Case | Draft references | Baseline matches / predictions | Contact matches / predictions | Fusion matches / predictions | Refined matches / predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 19 | 3 | 0 / 0 | 1 / 1 | 1 / 1 | 1 / 1 |
| 20 | 4 | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| 21 | 3 | 2 / 2 | 1 / 6 | 2 / 7 | 3 / 7 |
| 22 | 6 | 3 / 5 | 6 / 7 | 6 / 9 | 6 / 6 |
| 23 | 3 | 0 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |

Refined recovers a parent-crossing origin connection in case 20, selects a better
localized observation of an existing path in case 21, and rejects three unsupported
or cropped-continuation proposals in case 22. All four remaining unmatched
predictions are in case 21. All detectors completed all five cases without failures.
The draft cases informed these changes and are development data, not a holdout.
See the [specific failure analysis and synthetic validation](refined_detection.md).

## Per-branch error review

Export scored assignments and nearest-contact diagnostic evidence without
rerunning detection or changing references:

```powershell
python scripts/review_detection_errors.py --report nifti_previews/improvement_refined/report.json --detector refined --output-dir nifti_previews/improvement_refined/review --previews
```

This writes `review.csv`, `review.json`, and optional CT preview pages. CSV rows
include matched measurement errors, every unmatched reference, and every
unmatched prediction. JSON retains candidate measurements and contact rejection
records. Input hashes must still match the saved evaluation report. A nearest
contact is a diagnostic clue; it is not automatically the same anatomical branch.
Baseline exposes aggregate rejection counts, so its exact per-reference failure
stage cannot always be recovered. Failed cases are listed separately.

Omit `--previews` for a backend-only export; preview rendering uses the existing
matplotlib visualization dependency and currently requires fusion/refined path
diagnostics. Previews show three-voxel slabs with projected paths, not a full 3-D
adjudication. Review all projections and the original CT before assigning anatomy.

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
