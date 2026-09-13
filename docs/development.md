# Development and visualization

CPU-only infrastructure for detecting direct daughter arteries of the supplied
parent-aorta mask. The backend validates input geometry and runs an experimental
detector. Its accuracy on real scans remains unmeasured. See the
[detection method and candidate review guide](artery_detection.md).

## Offline setup

The submitted archive must include the populated `vendor/wheels/` directory.
Judges can install every backend and visualization dependency without internet
access using one command:

```powershell
python scripts/install_offline.py
```

The bundled wheelhouse currently targets **CPython 3.14 on Windows x64**. Before
packaging for any other judging runtime, populate it on an internet-connected
machine with the same operating system, architecture, and Python version:

```powershell
python scripts/download_wheels.py
```

The installer verifies its platform manifest, wheel SHA-256 hashes, and bundled
Mesa software OpenGL DLL hashes, then
passes both `--no-index` and `--find-links` to pip, so it cannot silently reach
PyPI during offline judging.

## Required run command

```powershell
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

For example, run the current valid baseline against a local case with:

```powershell
python run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output prediction.json
```

The baseline validates input geometry and writes candidate ostia, seeds 5 mm
along each traced branch, radii, and directions in physical LPS coordinates.
An empty `daughters` list means no candidates passed the current filters.

### Backend input loading

`backend.inputs.load_case(image_path, mask_path)` returns the CT and parent mask
as SimpleITK images on a shared 3-D grid. The evaluator and the compatibility
entrypoint `utils.case.load_case` use this loader.

- Compression is detected from file contents. Gzip data named `.nii`, including
  subjects 016-025, is read through a temporary `.nii.gz` copy that is removed
  after loading. Original dataset files are never rewritten or renamed.
- Ordinary inputs retain SimpleITK's native voxel values and physical geometry.
- If ITK rejects a nonorthogonal orientation, as in subject024, the loader first
  checks that the original image and mask shapes and physical affines match.
  It then resamples both onto one orthogonal grid covering their original
  physical extent. CT interpolation is linear with float32 output; mask
  interpolation is nearest-neighbor, preserving labels. Newly exposed space
  outside the source volume is filled with zero. The working grid can therefore
  have a different shape and origin from the source tensor shown in the viewer.
  A runtime warning reports this recovery.
- Recovery uses the coded NIfTI sform (or coded qform if no sform exists), converts
  RAS to LPS, and converts declared metres or microns to millimetres. Unspecified
  units follow ITK's existing millimetre assumption; the source unit is recorded
  in `branchseed_source_spatial_unit` metadata on resampled images. A missing,
  nonfinite, or singular recovery affine is rejected. Geometry mismatches and
  unrelated reader errors remain errors.

Physical anatomy stays in the same LPS coordinate system during resampling.
Use `image.TransformIndexToPhysicalPoint((x, y, z))` on the **returned image** for
evaluator coordinates; its NumPy array uses `(z, y, x)` order. Indices from the
native `core.Scan` tensor must not be used as indices into a resampled image.
The resampling transform maps each output physical point into the original
voxel grid, following [SimpleITK's resampling conventions](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html#resampling).

## Dataset inspection

Inspect every `.nii` or `.nii.gz` file recursively. The script reads image
headers only, so it does not load full CT volumes into memory:

```powershell
python eda/inspect_nifti_dataset.py dataset
```

The report includes voxel shape, voxel spacing, approximate physical size in
millimetres, data type, subject, and whether the file is an `orig` image or a
`mask`. Export machine-readable records with:

```powershell
python eda/inspect_nifti_dataset.py dataset --output nifti_dimensions.csv
python eda/inspect_nifti_dataset.py dataset --output nifti_dimensions.json
```

## Synthetic test cases

Generate reproducible CT-like cases with a parent-aorta mask, visible daughter
tubes, and JSON ground truth for controlled algorithm tests:

```powershell
python scripts/generate_synthetic_cases.py --output synthetic_dataset --cases 10 --seed 2026
```

Each generated `subjectNNN/` contains `origNNN.nii`, `maskNNN.nii`, and
`truthNNN.json`. The mask contains only the parent tube; daughter ostia, 5 mm
seeds, radii, directions, and curved centerlines are recorded in the truth file.
Use `--min-daughters`, `--max-daughters`, `--curvature`, `--noise-std`, `--size`,
and `--spacing` to vary the test set while retaining physical-space vessel
dimensions. Daughter branches use separated launch angles and are rejected if
their centerlines would overlap.

## Voxel-intensity histograms

Generate one Matplotlib histogram per CT/original volume. Values include NIfTI
scaling and are typically Hounsfield units for calibrated CT scans:

```powershell
python eda/plot_intensity_histograms.py --dataset dataset --output-dir nifti_histograms
```

The y-axis uses a logarithmic scale by default. A dashed horizontal line marks
the number of nonzero voxels in the paired aorta mask, and a red overlay shows
the intensity distribution restricted to those mask voxels. The histogram range
is the full finite range of non-padded voxels; values at or below `-2048` are
treated as CT padding by default. Customize that threshold with `--padding-floor`
and the bin count with `--bins`; use `--linear-y` for a linear y-axis. Add
`--include-masks` to generate histograms for mask volumes too.

## Visualization

From the repository root, start the optional frontend visualization with:

```powershell
python -m streamlit run frontend/app.py
```

The dashboard lists NIfTI scans in the chosen dataset folder and its immediate
subject subfolders. Select a subject, CT file, and optional neighboring aorta
mask in **Case files**. With a 3-D CT and mask, **Branch detection** runs the selected
experimental algorithm (baseline by default) once per input revision and caches
its evaluator-format results. Choose an instance to emphasize or hide the branch
overlay. The main panel has **Simple View**, **Detailed View**, and **Results**
choices. Only the selected view runs, so opening Results does not start a VTK
render. **Results** places the evaluator JSON export above the
measurements. It displays each candidate's parent ID, physical ostium and
5 mm seed coordinates, radius in millimetres, and unit direction. An empty
detector result is reported explicitly.
These candidates are proposals, not verified anatomical labels.

The dashboard is titled **Find Your Daughter** and uses a dark, flat rose
palette with light text. A compact project logo appears above the sidebar controls.
The case summary and sidebar controls start near the top
of their panels. **Simple View** is the default renderer. It sends
a bounded sample of CT voxels and mask surface points in the NIfTI RAS display
geometry to a Canvas 2D viewer. Detected LPS points and directions are
converted to RAS for overlay: a blue dot marks the ostium, a teal arrow gives
the direction, and a pale teal ring marks the seed and estimated radius
perpendicular to that vector. The ring is a measurement glyph, not a segmented
vessel surface. Drag to orbit, Shift + drag to pan, scroll to zoom, double-click to
reset, or save the current view as PNG. Click a branch marker, arrow, or ring to
show its ostium, seed, radius, and direction. The displayed positions are
physical millimetres. These are candidate measurements for source-CT review,
not an anatomical diagnosis. Camera motion does not rerun Streamlit or render
a new server PNG. Window, level, sampling, and visibility changes
rebuild the sampled scene. This is a point preview, not volumetric CT rendering;
it uses neither WebGL nor external assets.

Choose **Detailed View** for the native CPU volume scene. Sidebar dropdowns group
volume contrast, camera, and slice-plane controls. These include minimum
intensity, opacity, MIP, CT sampling, mask focus, three orthogonal I/J/K voxel
indices with physical axis labels, independent 3-D plane visibility and opacity,
and a CT cut at K. Slice panels can be shown beside the volume view. I/J/K are
voxel indices for browsing, not physical-space detection coordinates. Each
slider or toggle change starts a new render automatically; download the image
as PNG when ready. The detector measurements are currently overlaid in the fast
view and listed in Results. Detailed View now draws the same blue ostium,
teal direction arrow, pale teal radius ring, and branch label over its VTK
volume and slice scene; its branch visibility and selection follow the shared
sidebar controls. When VTK is unavailable, the CPU preview retains those
branch markers.
A progress bar shows elapsed rendering time, and a completion message appears
when the image is ready. The dashboard keeps a VTK render process and its loaded
scan alive while the selected case, frame, and CT sampling limit stay the same.
Changing those settings rebuilds the scene and takes longer than camera or
display adjustments. Rendering still uses the CPU-only Mesa `softpipe` driver,
so higher CT sampling limits and large output images can take longer than a GPU.
In Detailed View, the browser displays a VTK-rendered image; rotation uses
the camera sliders.
If VTK cannot create its offscreen context, the dashboard explicitly reports
the failure and shows the simpler CPU point preview instead.

On Windows x64, `vendor/mesa/` supplies OSMesa and `libglapi` from Mesa3D
24.3.4. The dashboard's VTK subprocess uses Mesa `softpipe` for software
OpenGL, so the 3-D volume render can run on CPUs without a GPU or browser
WebGL. The normal Windows OpenGL path failed in our headless session, and
Mesa's `llvmpipe` driver exited with an illegal instruction; `softpipe`
rendered the full-size subject successfully. The subprocess changes no system
graphics settings and uses no network. The NIfTI RAS affine is used for display.

The existing NIfTI quick-look utility lives at `scripts/visualize_nifti.py`:

```powershell
python scripts/visualize_nifti.py --dataset dataset --output-dir nifti_previews
```

The utility detects gzip compression from file contents, including the compressed
`.nii` images and masks in subjects 016–025; renaming the dataset files is unnecessary.

## Scan data and tensor previews

`core` provides `Scan`, `ScanGeometry`, `ScanCase`, `PreviewOptions`, and
`PreviewMode`. `Scan.data` is a NumPy `float32` tensor with NIfTI scaling applied.
It retains the native voxel order `(i, j, k)` or `(i, j, k, frame)`; loading does
not transpose, reorient, or normalize it. `storage_dtype` records the source
datatype. Frame and slice access return views of the array.

```python
from core import Scan, ScanGeometry, ScanCase, PreviewOptions

case = ScanCase.from_nifti(
    "dataset/subject016/orig16.nii",
    "dataset/subject016/mask16.nii",
)
scan = case.image
print(scan.shape, scan.dtype, scan.geometry.spacing)
print(scan.data[150:156, 150:156, 49])  # Actual NumPy voxel values
slice_tensor = scan.slice(axis=2, index=49)  # Same as scan.data[:, :, 49]

case.export_preview(
    "nifti_previews/subject016_both.png",
    PreviewOptions(mode="both", axis=2, slice_index=49, patch_size=6),
)
scan.export_numpy("nifti_previews/subject016.npy")
```

You can also construct a scan from a NumPy array with
`Scan(data=array, geometry=ScanGeometry(affine_ras=affine, spatial_unit="mm"))`.
Use an actual voxel-to-RAS affine for the source scan. `ScanCase` checks image
and mask shapes, affines, units, and compatible frame counts before overlaying.
A 3-D mask can be shared across a 4-D image's frames.

The geometry stores the NIfTI RAS affine and the header's spatial units;
`unknown` units remain unknown. Array axes are identified by their orientation
codes, and are not assumed to be physical x/y/z axes. This model does not change
the evaluator's SimpleITK LPS coordinate conversion.

Choose a preview mode on the command line:

```powershell
# Image slices with red mask overlays (the default)
python scripts/visualize_nifti.py --dataset dataset/subject016 --mode image

# Native tensor heatmap and a numeric patch, alongside the image preview
python scripts/visualize_nifti.py --dataset dataset/subject016 --mode both

# Inspect a specific tensor slice and export full image/mask arrays
python scripts/visualize_nifti.py --dataset dataset/subject016 --mode tensor --axis 2 --slice-index 49 --patch-origin 150 150 --patch-size 6 --export-numpy
```

`--frame` selects a 4-D frame (default 0); `--show` also opens the exported plot.
`--slice-index` selects the tensor slice and the matching image panel; other
image panels use their central slices. Tensor heatmaps keep native row/column
order and the full slice's value range. The red box locates the numeric patch,
whose labels use six significant digits. The image panels transpose slices for
display and use voxel spacing for their aspect ratio.

PNG files use `_tensor` or `_both` suffixes for those modes. `--export-numpy`
saves all frames of the scaled image and mask arrays as `.npy` files, with no
display transformations. Load them with `numpy.load(path, allow_pickle=False)`.
These files contain voxel arrays only, without spatial metadata; retain the
original NIfTI or `Scan.geometry` when physical geometry is needed. Exports
default to the ignored `nifti_previews/` directory. Matplotlib is imported by
the core only when a preview is requested.

## Native 3D volume viewer

Add `--detect` to inspect experimental detector candidates with linked CT slices,
branch selection, and radius rings. See the [detector visualization guide](artery_detection.md#run-and-inspect)
for controls, coordinate handling, and screenshot examples.

Open a scan in the VTK desktop viewer; no browser or web server is involved:

```powershell
python scripts/view_nifti_3d.py --image dataset/subject016/orig16.nii
```

The viewer detects a single neighboring mask automatically. Use `--mask path`
to select one explicitly, or `--no-mask` to show only the scan. It accepts both
ordinary NIfTI files and gzip data mislabeled as `.nii`.

The large viewport renders the CT volume with adjustable transparency and a red
surface of the supplied mask. Three smaller views show full-resolution slices
with red mask overlays. Image spacing and the full NIfTI affine position all
layers together, including oblique scans. The labels name native voxel axes;
they do not assume the scan is already in standard anatomical orientation.

| Control                      | Action                                                                           |
| ---------------------------- | -------------------------------------------------------------------------------- |
| Drag in 3D / mouse wheel     | Rotate / zoom                                                                    |
| I, J, K sliders              | Move the three slice positions                                                   |
| Wheel over a slice           | Step through that axis                                                           |
| Click a slice                | Move the other slice positions to that voxel                                     |
| Window level / width sliders | Adjust CT contrast and the volume transfer function                              |
| Minimum intensity slider     | Hide volume and slice pixels below the selected intensity                        |
| Volume opacity slider        | Reveal or hide tissue inside the volume                                          |
| V / M / P                    | Toggle the CT volume, mask surface, or slice planes in 3D                        |
| C                            | Toggle a CT cutaway, keeping voxel K at or below the K slider                    |
| B                            | Switch between composite volume rendering and maximum intensity projection (MIP) |
| F / R                        | Focus on the mask / reset the 3D camera                                          |
| S / Q or Escape              | Save a PNG / close the window                                                    |

This is a volumetric display: transparency integrates tissue along viewing rays,
MIP emphasizes the brightest voxels, and slice views expose interior detail.
The mask surface comes only from supplied foreground voxels; no new segmentation
is performed. Its surface closes at scan boundaries, including cropped ends.

The CT render uses a sampled grid capped at 192 voxels on its longest axis by
default. Increase `--max-dimension` for more detail or lower it for responsiveness.
This sampling can omit small CT structures; slice views and mask geometry always
use the full-resolution input. The CPU ray caster uses one thread for repeatable
rendering. It does not require GPU volume computation; the native window still
needs a working OpenGL display driver. Rendering is for visual inspection, with
the evaluator's SimpleITK coordinate path unchanged.

`--frame` selects a 4-D frame. `--window`, `--level`, and `--opacity` set initial
display values (defaults 400, 40, and 0.12). `--min-intensity` hides lower-valued
volume samples and renders lower-valued slice pixels black. Window/level and the
threshold use scaled voxel values, which are HU when the CT is calibrated. Unknown
spatial units remain unknown.

```python
from core import ScanCase, VolumeViewOptions

case = ScanCase.from_nifti("dataset/subject016/orig16.nii", "dataset/subject016/mask16.nii")
case.show_3d(VolumeViewOptions(max_dimension=192, window=400, level=40))
# A standalone Scan also has show_3d(), optionally accepting mask=another_scan.
```

Screenshots default to `nifti_previews/<case_id>_3d.png`. For a render that exits
without opening an interactive window:

```powershell
python scripts/view_nifti_3d.py --image dataset/subject016/orig16.nii --offscreen --screenshot nifti_previews/subject016_3d.png
```

VTK is pinned in `requirements-frontend.txt`, bundled in `vendor/wheels`, and
included in the offline installer. The Windows x64 Mesa DLLs are bundled in
`vendor/mesa/` with pinned hashes. Importing `core` does not import VTK or open
a window. The native viewer retains its interactive controls; Streamlit reuses
its VTK volume scene as the main view.

## Verification

Run the backend, scan model, and preview tests with:

```powershell
python -m unittest discover -s tests -v
```

## Layout

```text
backend/                  evaluator-facing Python package
core/                     NumPy scan models and optional preview rendering
desktop/                  native VTK volume viewer
docs/                     detailed development and visualization guidance
eda/                      dataset inspection utilities
frontend/app.py           optional Streamlit entrypoint
scripts/                  visualization and offline dependency tooling
vendor/wheels/            bundled pip wheels for offline installation
vendor/mesa/              bundled Windows x64 software OpenGL libraries
run.py                    required evaluator entrypoint
requirements-backend.txt  backend runtime dependencies
requirements-frontend.txt visualization dependencies
```
