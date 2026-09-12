# Find Your Daughter

Branchseed Challenge submission. The backend now proposes direct daughter arteries using an experimental CPU baseline. Real-scan detection accuracy is not yet measured; see the [detection method, validation, and candidate previews](docs/artery_detection.md).

The bundled offline wheels target CPython 3.14 on Windows x64. From the repository root, install dependencies:

```text
python scripts/install_offline.py
```

Run a case:

```text
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Run a local case after setup:

```powershell
python run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output prediction.json
```

The baseline validates input geometry and writes candidate ostia, seeds 5 mm
along each traced branch, radii, and directions in physical LPS coordinates.
An empty `daughters` list means no candidates passed the current filters.

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

Start the Streamlit frontend with:

```powershell
python -m streamlit run frontend/app.py
```

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

Use `--min-intensity VALUE` to set the initial cutoff. The viewer also provides
a live Minimum intensity slider to hide lower-intensity volume samples and
render lower-intensity slice pixels black.

See [3D viewer controls, options, and screenshots](docs/development.md#native-3d-volume-viewer) or the [development guide](docs/development.md) for details.
