# Development and visualization

CPU-only infrastructure for detecting direct daughter arteries of the supplied
parent-aorta mask. Detection is not implemented yet; the current backend
validates input geometry and writes a schema-valid JSON envelope with an empty
`daughters` list.

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

The baseline validates that the volumes share the same 3-D physical grid and
writes the required schema. Daughter detection is the next algorithm layer;
until it is implemented, the output contains an empty `daughters` list.

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

## Visualization

Start the Streamlit frontend with:

```powershell
python -m streamlit run frontend/app.py
```

The dashboard lists NIfTI scans in the chosen dataset folder and its immediate
subject subfolders. Select a subject, CT file, and optional neighboring aorta
mask. It validates image/mask geometry, then renders the existing VTK volume
scene into the dashboard. Sidebar controls adjust the camera, opacity,
window/level, sampling limit, MIP, and a voxel-K cutaway. The **Orthogonal
slices** controls display the native VTK slice panels beside the 3-D view.
Axis I, J, and K sliders move the three cross-sections; **Show slice planes in
3D** also places those sections in the volume scene. These are voxel indices
for browsing, not physical-space detection coordinates. Each slider or toggle
change starts a new render automatically; download the image as PNG when ready.
A progress bar shows elapsed rendering time, and a completion message appears
when the image is ready. The dashboard keeps a VTK render process and its loaded
scan alive while the selected case, frame, and CT sampling limit stay the same.
Changing those settings rebuilds the scene and takes longer than camera or
display adjustments. Rendering still uses the CPU-only Mesa `softpipe` driver,
so higher CT sampling limits and large output images can take longer than a GPU.
The browser displays a VTK-rendered image; rotation uses the camera sliders.
If VTK cannot create its offscreen context, the dashboard explicitly reports
the failure and shows the simpler CPU point preview instead.

On Windows x64, `vendor/mesa/` supplies OSMesa and `libglapi` from Mesa3D
24.3.4. The dashboard's VTK subprocess uses Mesa `softpipe` for software
OpenGL, so the 3-D volume render can run on CPUs without a GPU or browser
WebGL. The normal Windows OpenGL path failed in our headless session, and
Mesa's `llvmpipe` driver exited with an illegal instruction; `softpipe`
rendered the full-size subject successfully. The subprocess changes no system
graphics settings and uses no network. The NIfTI RAS affine is used for
display; this viewer performs no artery detection.

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
display values (defaults 400, 40, and 0.12). Window/level use scaled voxel values,
which are HU when the CT is calibrated. Unknown spatial units remain unknown.

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
