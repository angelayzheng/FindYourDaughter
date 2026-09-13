# Evaluate detectors against the draft reference set

Run all registered algorithms with their default settings:

```powershell
python scripts/evaluate_detectors.py
```

This reads `eval_set/case_*/` and writes `nifti_previews/evaluation/report.json`,
`summary.md`, and separate prediction/diagnostic JSON files under `baseline/`,
`contact/`, `fusion/`, and `refined/`. This evaluation command performs no training or parameter search;
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

The [25-case annotation workspace](annotation_workflow.md) prepares the remaining
scans for review; it does **not** extend the labeled reference set. Unreviewed
workspace cases have unknown daughters and are rejected by this scorer rather
than treated as empty negative cases. The original five-case drafts below are
preserved with their existing limitations.

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
to their saved results before refinement; their default detection behavior and the default
selection remain unchanged. `refined` is a separate opt-in algorithm.
The command was:

```powershell
python scripts/evaluate_detectors.py --detector all --output-dir nifti_previews/improvement_refined
```

The local `nifti_previews/improvement_refined/report.json` records these historical metrics,
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

## Parameter tuning results (2026-09-13)

The original table above is retained. A separate search evaluated **114
configurations** across all five cases, ranked by balanced **F1** at the same
3 mm tolerance. All 570 case/configuration evaluations completed successfully.
The best three settings per detector and its default then ran on synthetic
development data (10 cases, 30 daughters). Selection requires synthetic F1 and
recall at least as good as that detector's default.

| Detector | Original F1 | Selected matches / 19 | Selected extras | Selected precision | Selected recall | Selected F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 37.0% | 8 | 6 | 57.1% | 42.1% | 48.5% |
| Contact | 57.1% | 9 | 1 | 90.0% | 47.4% | 62.1% |
| Fusion | 57.9% | 12 | 7 | 63.2% | 63.2% | 63.2% |
| Refined | 72.2% | 14 | 3 | 82.4% | 73.7% | **77.8%** |

| Saved configuration | Changes from that detector's original defaults |
| --- | --- |
| [Baseline](../configs/tuned_20260913/baseline.json) | `baseline.intensity_fraction=0.45` |
| [Contact](../configs/tuned_20260913/contact.json) | `contact.intensity_fraction=0.45` |
| [Fusion](../configs/tuned_20260913/fusion.json) | `contact.intensity_fraction=0.30`, `contact.min_vesselness=0.15` |
| [Refined](../configs/tuned_20260913/refined.json) | `contact.intensity_fraction=0.30`, `contact.min_vesselness=0.15` |

Refined is the strongest selected configuration in this search. It recovers
case 19 / `branch_001`, keeps every previously matched reference, and reduces
case 21 extras from four to three. Its per-case matches / predictions are
**2/2, 2/2, 3/6, 6/6, 1/1** for cases 19 through 23. Lowering the brightness
threshold from 256.55 to 232.90 HU recovers the weak path in case 19;
requiring stronger tube evidence
limits extras. Five references remain missed, chiefly involving absent or
displaced contact proposals and short paths ending at forks.

The raw draft-F1 leader is a different [Refined configuration](../configs/tuned_20260913/refined_draft_f1.json):
`contact.min_vesselness=0.15`, `refined.minimum_natural_sections=1`.
It achieves **78.8% F1**, 92.9% precision, 68.4% recall (13 matches, 1 extra),
but drops synthetic development recall from 28/30 to 27/30. It is recorded as
an experimental alternative and fails the declared selection guard. Requiring
natural closure can reject a real branch next to the parent or another bright
structure. Contact's F1 improvement also costs one draft match; F1 does not
guarantee that precision and recall both increase.

Fresh synthetic validation used **16 cases / 46 daughters**, generated after
the settings were frozen. Refined and Fusion each retain **45/46 matches and
1 extra** (97.8% F1). Contact improves from **42/46 with 1 extra** to
**45/46 with 0 extras** (94.4% to 98.9% F1). Baseline regresses from **45/46
to 42/46**, with one extra in each (97.8% to 94.4% F1); its draft-selected
configuration is not a general improvement. The recorded selection was not
changed using this validation set. All 128 default/selected case runs completed.

The selected Refined configuration also completed all 25 local scans with four
logical CPUs: mean inference **2.43 s**, maximum **6.33 s**, process peak working
set **618.8 MiB**. This local resource check excludes imports and output writes;
loading is recorded separately. It does not establish official judge runtime.

See the [complete 114-trial record and fresh synthetic validation](experiments/tuning_20260913.md),
its [machine-readable results](experiments/tuning_20260913.json), and
[parameter usage and reproduction commands](parameter_tuning.md). These are
best observed settings within this grid on draft development annotations,
not a global optimum or a real held-out accuracy estimate. Existing default
predictions were verified identical across all 20 detector/case pairs before
and after exposing parameters. Tuned behavior requires an explicit `--config`.

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

For reproducible parameter search with preserved trials and unchanged defaults,
see [parameter tuning](parameter_tuning.md). A saved `--config` can also be passed
to this evaluation command; effective settings are recorded in `report.json`
under `configurations` and in each detector's diagnostics.

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
