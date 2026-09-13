# Experimental contact detector

`backend/contact_detection.py` adds a second, CPU-only detector inspired by the
three papers in `papers/`. Select it with `--detector contact`. The original
algorithm in `backend/detection.py` is unchanged, and `baseline` remains the
default. Both return the same evaluator JSON schema and physical LPS coordinates.
The new method improves specific controlled tests; expert-validated real-scan
accuracy is still unmeasured. A [paired draft-reference evaluation](evaluation.md)
now covers the supplied five-case annotation set, with per-case tradeoffs.

## Run and inspect

Open the native VTK viewer with the new candidates:

```powershell
python scripts/view_nifti_3d.py --image dataset/subject001/orig1.nii --detect --detector contact
```

The title identifies the selected algorithm. The supplied parent surface, short
paths, ostia, 5 mm seeds, directions, and radius rings use the same working grid
as detection. Existing [selection and slice controls](artery_detection.md#run-and-inspect)
apply. `--detector` requires `--detect`; omit it to use the baseline. The default
contact screenshot is `nifti_previews/<case>_contact_detection_3d.png`.

Export evaluator JSON and static candidate overlays:

```powershell
python run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output nifti_previews/contact_prediction.json --detector contact
python scripts/preview_detection.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --detector contact
```

Static exports default to `nifti_previews/detection/contact/`; baseline exports
still default to `nifti_previews/detection/`. An explicit `--output-dir` takes
precedence; rerunning a case in that directory replaces its previous exports.
All generated outputs remain ignored by Git. No dependency changes or runtime
network access are required.

The diagnostic JSON records the detector name, thresholds, rejection counts,
and one record per candidate contact patch. A record includes its proposed
ostium and status, and, where available, traced path, stopping reason, observed
length, shape score, radius, and radius measurement method. Accepted records
include their output `instance_id`. These details stay outside evaluator daughter
objects. The viewer displays accepted candidates; rejected contact records can
be inspected in the diagnostic JSON.

## What was adapted from the papers

| Source | Applied here | Outside this implementation |
| --- | --- | --- |
| Danilov et al., 2016 ([paper 1](https://doi.org/10.3390/computation4030035)) | Use multiscale tube evidence before graph extraction; clean wall-adjacent responses from outer supported layers inward. | The paper's complete method and original voxel-distance parameters. Our near-wall band is explicitly in millimetres. |
| Riffaud et al., 2022 ([paper 2](https://doi.org/10.1007/s11517-022-02603-2)) | Inspect graph connections and fit a local physical axis for measurement. | Full-tree input, anatomical name assignment, and long distal length rules. |
| Tahoces et al., 2020 ([paper 3](https://doi.org/10.1007/s11517-019-02110-x)) | Estimate parent blood intensity, use bounded CT-compatible growth, and treat contacts with the supplied parent as candidate openings. | The complete two-phase growth procedure, named-vessel filters, and CPR interface. |

These are project-specific adaptations, not reproductions of the papers.
Thresholds below are provisional engineering settings, not published optimal
values or challenge eligibility definitions.

## Method

1. Crop a physical region around the supplied binary parent mask. Estimate blood
   intensity from its interior and background from a surrounding shell. Preserve
   the input CT and mask.
2. Compute multiscale Hessian tube evidence before segmentation. Start from strong
   exterior tube voxels and reconstruct through a more permissive CT intensity
   range. Limit growth to 12 mm from the parent and to a 2 mm neighborhood of
   tube evidence, allowing a 2 mm near-wall connection band. This bounds leakage
   through a narrow bridge into adjacent enhanced structures.
3. Preserve exterior support, then retain successive near-wall layers only when
   adjacent to already supported voxels. Keep components touching the parent.
   This suppresses thin wall responses without requiring a whole daughter to
   travel monotonically away from the aorta.
4. Label distinct exterior contact patches. Choose a central opening voxel using
   distance to the combined parent-and-candidate lumen wall, then locate the
   interpolated half-mask interface. Using the combined lumen avoids mistaking
   the artificial cut at the parent boundary for a daughter side wall.
5. Skeletonize the local lumen and collapse connected junction voxels. Match
   contacts to outward roots. Use local endpoints of the supplied parent's
   skeleton and local surface normals to reject probable parent continuations,
   including curved ends. This is still a heuristic for terminal anatomy.
6. Follow observed paths for at most 10 mm or to the first sustained fork, pruning
   sub-2-mm spurs. Stop on cycles or another parent contact rather than arbitrarily
   choosing a tree through ambiguous connections. Reject paths without enough
   observed length, exterior CT support, or tube evidence.
7. Place the seed 5 mm along the path. Fit an anchored local axis to sample a
   perpendicular CT section. Estimate an area-equivalent radius from a closed
   lumen region; keep air from lowering the threshold into soft tissue. Retry a
   leaking section once with the supplied parent excluded. Reject a section that
   remains open instead of assigning a distance-transform fallback radius.

Voxel arrays are ZYX. Integer indices use `TransformIndexToPhysicalPoint`, and
fractional indices use its continuous-index counterpart. Arc lengths and fitted
axes use physical LPS coordinates. The exported unit direction is the normalized
ostium-to-seed vector; the fitted local tangent defines the radius plane. IDs
are stable for a given input and algorithm, but do not identify the same anatomy
across algorithms or parameter changes.

The registry is `backend.detectors.detect(image, mask, detector="contact")`.
For controlled development experiments, use:

```python
from backend.contact_detection import ContactOptions, detect_contacts

result = detect_contacts(image, aorta_mask, ContactOptions())
daughters = result.daughters()
```

`image` and `aorta_mask` must be matching scalar 3-D SimpleITK images, such as
those returned by `backend.inputs.load_case`. The latter retains the existing
compressed-file and nonorthogonal-grid recovery behavior.

| `ContactOptions` setting | Default | Meaning |
| --- | --- | --- |
| `margin_mm` | 15 | Parent bounding-box padding; at least the growth distance |
| `growth_mm` | 12 | Maximum exterior growth distance; at least 10 mm |
| `intensity_fraction` | 0.35 | Background-to-blood fraction for the lower threshold, floored at 40 CT units |
| `support_band_mm` | 2 | Near-wall band requiring retained outer support |
| `tube_seed` | 0.15 | Minimum tube response for reconstruction markers |
| `tube_reach_mm` | 2 | Maximum distance from tube markers outside the near-wall band |
| `min_vesselness` | 0.10 | Minimum mean path tube response |
| `min_radius_mm`, `max_radius_mm` | 0.8, 6 | Accepted measured radius range |

## Controlled comparison

The current generator was run on 10 cases with 30 daughters, seed 941, size
128 x 128 x 192, default spacing 0.8 x 0.8 x 1.0 mm, and other defaults. Both
algorithms were evaluated on those same files with the existing one-to-one
ostium matcher at a **development tolerance of 3 mm**:

```powershell
python scripts/generate_synthetic_cases.py --output tmp/contact-comparison --cases 10 --seed 941 --size 128 128 192
python scripts/evaluate_synthetic_detection.py --dataset tmp/contact-comparison --detector baseline --output tmp/contact-comparison/baseline_metrics.json
python scripts/evaluate_synthetic_detection.py --dataset tmp/contact-comparison --detector contact --output tmp/contact-comparison/contact_metrics.json
```

| Measurement | Baseline | Contact |
| --- | ---: | ---: |
| Matched / 30 | 27 | 27 |
| Extra candidates | 0 | 0 |
| Missed daughters | 3 | 3 |
| Mean ostium error (mm) | 0.70 | 0.63 |
| Mean seed error (mm) | 0.84 | 1.03 |
| Mean absolute radius error (mm) | 0.51 | 0.25 |

Position and radius errors are computed only over each algorithm's matched
branches, which are not identical sets. Contact improves radius error in this
comparison but worsens seed error and does not improve total recall. Its misses
are a short extracted path, a contact without a skeleton root, and a path failing
CT/parent support. This cohort informed development; it is not held out and is
not the official challenge scorer.

Additional tests cover recovery of a weak-contrast branch (150 CT units versus
300 in the parent) missed by the baseline, supported wall cleanup, radius
measurement near air, curved parent continuations, ambiguous graph connections,
and deferral of an early common-trunk fork. The original ten physical/anatomical
contract tests also run against the new detector. The baseline-default test
compares its exported daughters directly with the original implementation.

All 25 supplied CT/mask pairs completed with both algorithms, including the
compressed `.nii` inputs and subject024's resampled grid. In a local Windows /
Python 3.13.7 run restricted to four logical CPUs, contact averaged 1.21 seconds
for loading plus detection (maximum 3.08 seconds); baseline averaged 0.98 seconds
(maximum 2.55 seconds). The process running both methods peaked at 610 MiB
working set. Timings exclude imports, output writes, and visualization. These
are local resource checks, not judge-hardware measurements; the bundled target
remains CPython 3.14 / Windows x64, which was not executed here.

Contact returned 104 proposals across the real scans versus baseline's 111.
Contact returned none for subjects 007, 016, and 024, whereas baseline returned
6, 5, and 7 respectively. No real branch truth was available to score either
set; fewer proposals or an empty result must not be interpreted as improvement.

## Limits and next work

- Tube evidence and CT thresholds cannot distinguish arteries from all veins,
  enhanced organs, calcium, or segmentation artifacts. Weak contrast and coarse
  resolution can erase valid openings. The method assumes HU-like enhanced CT.
- Distinct ostia with distinct contact patches remain separate. Touching patches
  may merge before graph extraction; rejecting an ambiguous connection can also
  discard a real daughter. Skeleton shortening can miss a visible thick branch.
- A common trunk that forks before the required 5 mm seed is still deferred as
  `early_bifurcation`; downstream daughters are not substituted for that trunk.
- Local cap rejection is approximate. The terminal iliac division and fragmented
  or extensive parent masks still need reviewed examples; reliable exclusion is
  not established. The method does not infer missing anatomy beyond the scan.
- Radius measurement is sensitive to partial volume, touching structures, local
  orientation, and CT thresholds. Rejecting an open section can lower recall.

The next useful step is to annotate real eligible ostia and proximal paths,
compare false positives and misses, and evaluate seed placement separately from
opening detection. Candidate counts on the supplied scans are not accuracy.
