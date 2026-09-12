# Experimental artery detection

`backend/detection.py` implements a deterministic, CPU-only baseline for proposing
arteries arising directly from the supplied parent mask. `run.py` now exports
these candidates instead of an intentionally empty scaffold. This is a starting
point for annotated evaluation: the real scans still contain questionable
candidates, and their detection accuracy has not been measured.

## Run and inspect

The evaluator command is unchanged:

```powershell
python run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output prediction.json
```

Export candidate overlays for manual review with Matplotlib, without a browser:

```powershell
python scripts/preview_detection.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output-dir nifti_previews/detection
```

The preview command writes `<case>_prediction.json`, `<case>_diagnostics.json`,
and numbered `<case>_candidates_XX.png` pages. Each candidate has three native-grid
views: red is the supplied parent boundary, yellow is the proposed ostium, and
cyan shows the projected short centerline and the direction to its seed. The CT
and mask use a narrow maximum-intensity slab at the ostium; the projected path
can extend outside that slab. Plot axes are distances along the working voxel
grid, not anatomical LPS coordinates. Inspect all three projections together.
Old numbered pages for that case are replaced when exporting to the same folder.

JSON coordinates are physical LPS millimetres. Diagnostics include thresholds,
rejection counts, short centerlines, and whether each radius used a perpendicular
section or the distance-transform fallback. Centerlines and diagnostic fields
are separate from the evaluator's JSON schema. No candidates produces an empty
list and an explanatory preview page. Generated files stay out of Git.

The preview uses the backend's returned grid, including the shared resampled
grid for subject024. This differs from the native tensor in `core.Scan`; always
use physical coordinates when comparing them. See [input loading](development.md#backend-input-loading).

## Method

1. Crop around the supplied parent mask with a 15 mm margin. Estimate blood
   intensity inside the parent and background intensity in an outer shell.
2. Smooth the CT and select a provisional enhanced-lumen intensity range within
   12 mm of the parent. Retain connected components touching the supplied parent.
   The parent mask itself is used unchanged, rather than predicted from CT.
3. Skeletonize this local lumen in 3-D and construct a graph. Find graph edges
   crossing from the parent to outside it; these are candidate openings. Merge
   neighboring junction voxels to reduce spurious voxel-grid forks. The thinning
   uses scikit-image's [Lee skeletonization](https://scikit-image.org/docs/0.26.x/api/skimage.morphology.html#skimage.morphology.skeletonize).
4. Reject likely cropped ends using the parent's principal axis and local surface
   normal. Trace outward from each opening, pruning short spurs and stopping at
   10 mm or the first downstream fork. Separate openings remain separate roots;
   a common trunk with sufficient visible length produces one candidate.
5. Reject paths that run along the parent wall, leave the observed lumen, or have
   weak tubular shape. The latter uses a multiscale Hessian response through
   SimpleITK's [ObjectnessMeasureImageFilter](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ObjectnessMeasureImageFilter.html).
   This response is a shape score, not an artery probability.
6. Interpolate the seed at 5 mm of centerline **arc length**, estimate radius from
   lumen area in a plane perpendicular to the local path, and normalize the
   vector from ostium to seed. If a closed local section cannot be isolated,
   use a conservative distance-transform radius and record that fallback.

Skeleton indices are converted with SimpleITK's
`TransformIndexToPhysicalPoint`; interpolation and distances then use physical
space. Output IDs are deterministic `branch_001`, `branch_002`, etc., ordered by
ostium coordinates. The detector does not assign anatomical names or assume a
fixed branch count. SciPy and scikit-image are pinned backend dependencies with
wheels included in the offline bundle; runtime processing needs no network.

For development, `detect_daughters(image, aorta_mask, DetectionOptions(...))`
accepts these provisional settings:

| Option | Default | Meaning |
| --- | --- | --- |
| `margin_mm` | 15 | Parent bounding-box padding; minimum 12 mm |
| `intensity_fraction` | 0.55 | Fraction from estimated background to parent blood for the lower threshold; also floored at 60 CT units |
| `min_radius_mm` | 0.8 | Smallest accepted estimated radius |
| `max_radius_mm` | 6.0 | Largest accepted estimated radius |
| `min_vesselness` | 0.1 | Minimum mean tubular-shape response |

These are algorithm filters, not challenge eligibility definitions. The method
assumes scaled CT intensities comparable to HU and visible contrast enhancement.

## Development checks

Ten dedicated synthetic anatomy tests cover physical coordinates, anisotropic
spacing and rotated/reflected grids, unit directions, stable unique instances,
empty cases, detached structures, cropped ends, nearby independent openings,
common trunks, curved paths, and mismatched geometry:

```powershell
python -m unittest discover -s tests -v
```

An additional local check used 10 cases from the project's tube generator:

```powershell
python scripts/generate_synthetic_cases.py --output tmp/branchseed/synthetic --cases 10 --seed 941 --size 128 128 192
python scripts/evaluate_synthetic_detection.py --dataset tmp/branchseed/synthetic --output tmp/branchseed/synthetic_metrics.json
```

With one-to-one ostium matching within a **development tolerance of 3 mm**, all
30 generated daughters matched, with no extra candidates or missed daughters.
Mean absolute errors were 0.48 mm for ostium position, 0.66 mm for seed position,
and 0.06 mm for radius. These simple enhanced tubes informed development and are
not a held-out validation set or the official challenge scorer. This result
does not establish performance on patient scans.

All 25 supplied real CT/mask pairs completed loading and detection, including
compressed `.nii` files and subject024. On this Windows machine with Python
3.13.7 and process affinity limited to four logical CPUs, mean load-and-detect
time was 1.00 s, maximum 2.52 s, and peak process working set about 613 MiB.
Timing excludes interpreter imports, JSON disk writes, and plotting. This is a
local resource check, not a measurement on judge hardware. The bundled target
remains CPython 3.14 / Windows x64; its dependency resolution is checked offline,
but Python 3.14 execution has not been tested on this machine.

## Known limitations and next steps

- Bright veins, calcification, noise, and adjacent blood pools can join the
  segmented lumen and create false branches. Low enhancement can hide real ones.
  Several noisy/coarse scans are visibly less reliable than the clean tube tests;
  subjects 017, 019, and 020 currently return no candidates. This does not imply
  that they contain no eligible arteries.
- Skeleton topology depends on thresholding and resolution. Narrow connections
  can disappear, neighboring vessels can merge, and complex loops are not
  explicitly resolved. Subvoxel openings and tangential branches can be missed;
  the seed currently must clear the parent by approximately 2 mm.
- End rejection is an approximate global-axis rule. Strong curvature and
  disconnected parent fragments need local endpoint analysis. Exclusion of the
  terminal iliac division is not yet reliable, and the algorithm does not
  automatically restrict a larger supplied mask to the abdominal region.
- A trunk that bifurcates before a 5 mm seed can be placed is currently deferred
  (`early_bifurcation`) rather than choosing one of its downstream branches.
  Such common trunks need an explicit challenge-compatible handling policy.
- Radius estimates are sensitive to partial volume and neighboring enhanced
  structures, particularly when the distance-transform fallback is used.

The next useful evidence is a small reviewed set of real ostia, seeds, and
eligible branches spanning both the cleaner and noisier scans. Use it to measure
misses and false positives before tuning thresholds. The main algorithmic next
steps are local parent-centerline endpoint classification and more selective
lumen separation near touching vessels and blood pools.
