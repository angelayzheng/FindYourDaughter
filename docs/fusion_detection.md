# Experimental fusion detector

`fusion` implements the improvement notes' shared candidate pipeline as a new
algorithm. The original `baseline` and `contact` implementations are unchanged,
and the default evaluator still uses `baseline`. Fusion runs both sources,
measures their proposals, groups duplicate observations, and selects one path
per group. It is a deterministic CPU algorithm, not a trained model.

## Run

```powershell
python scripts/view_nifti_3d.py --image dataset/subject001/orig1.nii --detect --detector fusion
python run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output nifti_previews/fusion_prediction.json --detector fusion
python scripts/preview_detection.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --detector fusion
python scripts/evaluate_detectors.py --detector all --output-dir nifti_previews/evaluation_fusion
```

The native viewer retains its existing controls and labels the selected method.
Fusion screenshots default to `nifti_previews/<case>_fusion_detection_3d.png`;
static previews default to `nifti_previews/detection/fusion/`. The evaluation
registry includes baseline, contact, fusion, and the separate
[refined detector](refined_detection.md). Fusion's default detection behavior is unchanged.
Optional [parameter configurations and tuning results](parameter_tuning.md) are
available separately. Generated predictions remain ignored.
No dependencies, network access, or changes to the evaluator JSON are required.

## Evidence and selection

`backend/candidates.py` defines an internal `Candidate`: a source observation,
its physical branch geometry, measurements, group membership, status, rejection
reasons, and selected/output IDs. `backend/fusion_detection.py` implements:

1. Collect accepted source proposals with their provenance. Preserve contact's
   rejected contact records separately, including early common trunks. Fusion
   cannot recover an opening that neither source proposed.
2. Recheck candidates in a parent ROI padded by 15 mm. Sample CT and parent-mask
   support between 2 and 5 mm along the path; record wall distances, surface
   distance, local normal alignment, vesselness, and parent-end distance. Use
   contact's local parent skeleton and surface-normal cap rule for both sources.
   A path returning into the parent or lacking sufficient support is rejected.
3. Measure perpendicular CT sections at 4, 5, and 6 mm where that much path was
   observed. Record closure, area, radius, nearby-ring contrast, coefficient of
   variation, and expansion ratio. A radius over 6 mm at 6 mm, together with
   more than a twofold radius increase from 4 mm, flags a narrow bridge opening
   into a broad pool. This is a provisional geometric guard, not tissue classification.
4. Pair baseline/contact observations one to one when origins are within 2.5 mm,
   directions have cosine at least 0.8, and corresponding path points at 1, 3,
   and 5 mm are close. The path allowance is the larger of 1.5 mm and the smaller
   measured radius. This accommodates distinct medial traces within a broad
   lumen. Assignment prevents transitive grouping from swallowing multiple
   separate roots from the same detector; nearby distinct narrow tubes stay separate.
5. Prefer observations with stable section measurements, then rank within a
   group using vesselness, CT/parent support, and section closure. Select one
   actual source path; do not average paths across anatomy. Place its seed at
   5 mm arc length and normalize its ostium-to-seed direction. Use the newly
   measured 5 mm radius when valid, otherwise retain the source estimate and
   explicitly mark its uncertainty in diagnostics.

Both sources receive only CT and the supplied binary parent mask. No annotation
coordinates, labels, confidence tags, per-case thresholds, or reference counts
enter detection. Array indexing is ZYX; all matching, tracing, and output geometry
are physical SimpleITK LPS millimetres.

The heuristic review score is `0.5 * vesselness + 0.25 * supported_fraction +
0.25 * closed_section_fraction`. It only ranks eligible observations inside a
duplicate group, not candidates across the scan. It is not an artery probability.
`stable_sections` means a closed seed section, at least two closed sections, and
radius coefficient of variation at most 0.35. Instability alone lowers confidence;
it does not automatically erase a borderline candidate.

`FusionOptions` exposes `duplicate_ostium_mm=2.5`, `duplicate_path_mm=1.5`,
`minimum_support_fraction=0.8`, `max_radius_cv=0.35`, and
`max_expansion_ratio=2.0`. Source `DetectionOptions` and `ContactOptions` retain
their own defaults. Settings and both source configurations are exported.

Diagnostics include all candidate records, duplicate/rejection reasons, selected
IDs, source counts and rejection counts, section evidence, and the original
contact records. A common trunk ending in a fork before a valid 5 mm seed is
retained as `needs_seed_policy` under `deferred_common_trunks`. It is not exported
as two downstream daughters or assigned an invented seed. Baseline only exposes
aggregate rejection counts, so not every baseline rejection has a recoverable path.

## Measured comparison

All comparisons use the existing one-to-one matcher with a fixed **3 mm** ostium
tolerance. Baseline tests and a draft-set report were saved before implementation.
The existing seed-941 synthetic cohort and all five draft cases are development
data. Parameters were frozen before generating/scoring a fresh seed-4271 cohort;
no changes were made in response to that holdout result.

| Dataset | Baseline matches / extras | Contact matches / extras | Fusion matches / extras |
| --- | ---: | ---: | ---: |
| Development tubes: 10 cases, 30 daughters, seed 941 | 27 / 0 | 27 / 0 | 28 / 0 |
| Held-out tubes: 12 cases, 33 daughters, seed 4271 | 30 / 0 | 31 / 0 | 33 / 0 |
| Draft real set: 5 cases, 19 references | 5 / 3 | 10 / 6 | 11 / 8 |

For the draft set, "extras" means unmatched predictions, not established false
positives. Fusion increases draft recall but reduces agreement precision from
contact's 62.5% to 57.9%. Neither annotations nor unmatched candidates have expert
adjudication. See the [per-case evaluation report](evaluation.md) for context.

On held-out tubes, mean ostium/seed/radius errors in mm were
0.65/0.79/0.50 for baseline, 0.57/0.96/0.37 for contact, and 0.58/0.87/0.43 for
fusion. Errors describe each method's matched subset. Improved recall does not
mean every measurement improved; fusion radius error is higher than contact's.
Generated tubes are still a limited distribution, not a clinical holdout.

Reproduce the fresh cohort with:

```powershell
python scripts/generate_synthetic_cases.py --output tmp/fusion-validation/heldout --cases 12 --seed 4271 --size 128 128 192
python scripts/evaluate_synthetic_detection.py --dataset tmp/fusion-validation/heldout --detector fusion --output tmp/fusion-validation/heldout_fusion.json
```

Spacing is the generator default (0.8, 0.8, 1.0 mm); all other generator settings
are defaults. Substitute `baseline` or `contact` and distinct output names for
comparison. The local `tmp/fusion-validation/frozen_configuration.json` records
the split, seeds, tolerance, options, and implementation hashes.

All 25 supplied scan pairs also completed, including subject024's resampled grid.
On this Windows / Python 3.13.7 machine with process affinity restricted to four
logical CPUs, mean load-and-fusion time was 2.51 seconds, maximum 6.67 seconds,
and peak process working set 621 MiB. Timing includes both nested source
detectors but excludes imports, output writes, and visualization. This is a
local resource check, not a judge-hardware measurement; the bundled CPython 3.14
target was not executed here. The full 110-test suite and native offscreen
viewer check passed.

## Remaining limitations

- Union of source proposals can add false positives. Fusion adds measurements
  and grouping; it does not independently establish artery identity.
- Duplicate association can still confuse touching origins. On draft case 21,
  fusion groups two observations of a nearby path and selects the contact origin
  just outside the 3 mm match tolerance. That choice loses a baseline reference
  match while preventing an extra duplicate. This ambiguity needs reviewed data.
- Six orthogonal preview pages covering every unmatched fusion prediction and
  unmatched draft reference were inspected. Several extra paths follow the wall
  or lie near bright structures; several missed draft guides have visible CT
  support. These are review priorities, not expert relabeling. Diagnostic review
  does not justify changing the reference inventory to favor a detector.
- A valid opening missed by both sources stays missed. Multiple intensity
  hypotheses, better wall-contact separation, early-fork seed policy, and reliable
  terminal-iliac exclusion remain open work.
- Broad-pool rejection, cap classification, and radius confidence are heuristics.
  Unusual anatomy, partial volume, and scan boundaries can still defeat them.

Targeted tests cover duplicate pairing, preservation of nearby ostia, broad-lumen
traces, thin bridges into pools, weak contrast with noise, coarse through-plane
spacing, early trunks, source provenance, and section evidence. The original
ten physical/anatomical contract tests also run against fusion.
