# Refined detector and failure analysis

`refined` is a separate experimental algorithm. The baseline, contact, and fusion
implementations, their settings, and the default `run.py` behavior are unchanged.
Their five draft-case prediction JSON envelopes were compared before and after
this addition and are identical. The new algorithm uses only CT and the supplied
parent mask, runs offline on CPU, and exports the existing six daughter fields.

```powershell
python run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output nifti_previews/refined_prediction.json --detector refined
python scripts/evaluate_detectors.py --detector refined --output-dir nifti_previews/evaluation_refined
```

The five real cases are development data with incomplete, expert-review-pending
annotations. Counts below measure draft agreement, not validated anatomical
accuracy. Reference IDs and prediction IDs are separate local namespaces.

## What the error review found

Before changing detection, all 113 existing tests passed and all three original
detectors were saved under `nifti_previews/improvement_before/`. Six orthogonal
preview pages covered every Fusion disagreement. The numerical investigation
then traced failures through contact records, mask support, CT samples, section
measurements, and duplicate selection.

| Finding | Evidence before refinement | New behavior |
| --- | --- | --- |
| Invalid origin-to-path chord, case 20 reference `branch_003` | Contact picked a point on another part of the wall patch. Its connection to the skeleton ran through parent voxels at 1-2 mm, so the entire proposal was rejected as `unsupported_path`. | Reanchor that short chord at its measured final parent exit. The new origin is 0.09 mm from the draft guide; the seed is 0.62 mm away and inside the matched label. |
| Duplicate selection displaced the ostium, case 21 reference `branch_003` | Fusion chose Contact's origin 3.07 mm from the reference over Baseline's 2.25 mm observation of the same proximal lumen. One Contact section needed the parent removed to appear closed. | Prefer naturally closed sections within an already matched duplicate group. Select the existing Baseline path; do not move landmarks toward annotations or change the 3 mm tolerance. |
| Parent continuation at the inferior crop, case 22 Fusion prediction `branch_001` | The origin lay about 13.5 mm from its parent-skeleton endpoint, beyond the old 10.7 mm cap neighborhood. The CT preview shows a continuation across the planar parent cut. | Measure the skeleton's retreat to the observed parent boundary along its local tangent; retain the existing normal-alignment cap test. |
| Unclosed sections, case 22 Fusion predictions `branch_005` and `branch_009` | Both had no closed CT lumen section at 4, 5, or 6 mm, even after excluding the parent. Fusion exported distance-transform radius fallbacks. Previews place these paths near broad bright structures. | Retain the proposals and evidence in diagnostics, but reject candidates with no closed proximal section. This is a geometric rejection, not a tissue diagnosis. |

Baseline never generated an accepted proposal for cases 19 and 20. Its aggregate
diagnostics cannot attribute every missed reference to a specific rejected root.
Contact adds useful candidates, but its single contact-patch representative can
be unrelated to the skeleton path, and strict CT checks discard borderline
paths. Fusion combines accepted sources; it cannot recover an opening that both
sources discard. Refined makes the bounded chord repair above in addition to
reviewing the accepted source proposals.

## Remaining disagreements

The following are all six missed draft references after refinement. A distant
nearest contact is only a clue; it does not establish the exact anatomical cause.

| Case / reference | Observed failure evidence | Remaining work |
| --- | --- | --- |
| 19 / `branch_001` | Contact origin agrees within 0.001 mm, but interpolated CT is 250.7-252.5 HU at 2.5-3 mm along the path, below its 256.5 HU threshold. | Address partial volume or path centering with independent weak-contrast fixtures; changing the origin cannot fix this intensity failure. |
| 19 / `branch_003` | Nearest contact is 3.28 mm away and stops at a fork after 3.80 mm. The draft notes also flag calcification and possible early branching. | Review graph topology and the common-trunk seed policy; no downstream daughter is substituted for the trunk. |
| 20 / `branch_001` | No contact within 3 mm; nearest is 9.89 mm away with a roughly 9.6 mm initial chord crossing the parent. | Improve contact localization/generation. The bounded repair deliberately refuses this long connection. |
| 20 / `branch_004` | No contact within 3 mm; the nearest original patch is 4.11 mm away and leads toward the path now matched to `branch_003`. | Review separation of neighboring contacts and preserve distinct outward paths. A merged contact is a hypothesis, not established anatomy. |
| 23 / `branch_001` | No contact within 3 mm; nearest is 9.71 mm away. That path has CT support failures through its first 5 mm. | Improve candidate generation for a small wall-adjacent origin; a distant rejected path cannot safely stand in for it. |
| 23 / `branch_003` | Nearest contact is 3.75 mm away and reports a fork after only 0.75 mm. Draft notes question diameter and separate origins versus a common trunk. | Adjudicate anatomy and improve short-junction handling without inventing a 5 mm seed. |

All four remaining extra predictions are in case 21:

| Refined prediction | Nearest draft ostium (mm) | Review evidence |
| --- | ---: | --- |
| `branch_001` | 63.75 | Wall-adjacent path: seed-to-parent-voxel-center distance is 1.58 mm and initial normal/direction cosine is 0.18. Wall following or an adjacent vessel is plausible. |
| `branch_002` | 49.32 | Near bright neighboring structures; mean vesselness is only 0.11. Independent origin remains unconfirmed. |
| `branch_003` | 30.41 | Projected path stays near the parent wall; direct connection and eligibility require review. |
| `branch_004` | 15.27 | Small wall-adjacent path with mean vesselness 0.13. A missed annotation and an incorrect candidate are both possible. |

The case's review notes explicitly say its inventory is not exhaustive. These
four predictions are not labeled confirmed false positives. Four new preview
pages cover every remaining refined disagreement; their paths and native-grid
axes are projected for review, while JSON coordinates remain physical LPS mm.

## Algorithm boundaries

`backend/refined_detection.py` runs the unchanged source detectors, reuses
Fusion's physical measurements and one-to-one source grouping, and adds:

1. Parent-cap reach measured from the local skeleton endpoint to the first
   observed mask exit along its tangent, with the existing three-voxel margin.
2. Repair only of an `unsupported_path` contact's initial parent-crossing chord.
   Reject chords longer than two physical voxel diagonals and origin corrections
   longer than one diagonal. Locate the half-mask exit by bisection, preserve the
   subsequent observed path, require at least 5 mm remaining, and recheck CT,
   parent exclusion, tubularity, and a closed measured seed section. Never relax
   intensity thresholds or bridge a distal gap.
3. Rejection when none of the observed 4/5/6 mm lumen sections closes. Partial
   section uncertainty can still remain; diagnostics retain each measurement.
4. Within existing duplicate groups, prefer stable, naturally closed sections
   before the inherited vesselness/support score. Select one source path without
   averaging coordinates. Sorting and instance IDs remain deterministic.

Limits remain: this cannot discover every missing contact, establish artery
identity, solve early common trunks, or reliably classify terminal iliac splits.
Stricter section checks can reject a real daughter near another bright structure.
Cap extension is still a heuristic for curved or fragmented parent masks.

## Validation

The [evaluation table](evaluation.md) records all metrics and per-case counts.
Refined improves draft agreement from Fusion's **11 matches / 8 extras** to
**13 matches / 4 extras**: precision **76.5%**, recall **68.4%**, F1 **72.2%**.
All 13 matched seeds lie in their matched draft labels. Direction error does
not improve overall, and numeric radius comparisons cover only two references.

After the code was frozen, a new synthetic cohort was generated and scored once.
No detection changes followed that result. All methods used the same fixed
3 mm matching tolerance and generator defaults:

| Dataset | Baseline matches / extras | Contact matches / extras | Fusion matches / extras | Refined matches / extras |
| --- | ---: | ---: | ---: | ---: |
| Development tubes, seed 941: 10 cases, 30 daughters | 27 / 0 | 27 / 0 | 28 / 0 | 28 / 0 |
| Fresh holdout tubes, seed 8063: 16 cases, 38 daughters | 38 / 1 | 37 / 1 | 38 / 1 | 38 / 0 |

Holdout mean ostium/seed/radius errors (mm) are **0.46/0.87/0.52** for Refined,
versus **0.46/0.91/0.51** for Fusion. The radius error is slightly worse. Generated
tubes are a limited distribution and do not establish clinical generalization.

```powershell
python scripts/generate_synthetic_cases.py --output tmp/model-improvement/holdout --cases 16 --seed 8063 --size 128 128 192
python scripts/evaluate_synthetic_detection.py --dataset tmp/model-improvement/holdout --detector refined --output tmp/model-improvement/holdout_refined.json
```

Spacing is the default (0.8, 0.8, 1.0 mm). Substitute the other detector names and
distinct output filenames for comparison. The local
`tmp/model-improvement/frozen_configuration.json` records split, seed, settings,
and backend hashes. Full predictions, diagnostic records, previews, and benchmark
files remain ignored by version control.

All 25 supplied scan pairs completed, including the existing subject024 geometry
handling. On this Windows / Python 3.13.7 machine, with process affinity limited
to four logical CPUs, mean load-and-detection time was **3.71 s**, maximum
**10.15 s**, and peak process working set **621 MiB**. This resource check ran
alongside other validation work; it is not an isolated speed comparison with
earlier benchmarks or a judge-hardware measurement. Timings exclude imports,
serialization, and visualization. The bundled CPython 3.14 target was not run.

Twenty dedicated checks cover the inherited geometry contract, early trunks,
weak contrast/noise, anisotropic spacing, broad bright pools, repaired chords,
cropped ends, open sections, and unchanged original predictions. The error-review
tests also ensure diagnostic proximity never replaces scored assignments.
The full 135-test suite, default and refined evaluator commands, `pip check`, and
`git diff --check` passed after the addition.
