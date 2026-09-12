# Branchseed Challenge

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

The installer verifies its platform manifest and wheel SHA-256 hashes, then
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

## Verification

Run the backend, scan model, and preview tests with:

```powershell
python -m unittest discover -s tests -v
```

## Layout

```text
backend/                  evaluator-facing Python package
core/                     NumPy scan models and optional preview rendering
eda/                      dataset inspection utilities
frontend/app.py           optional Streamlit entrypoint
scripts/                  visualization and offline dependency tooling
vendor/wheels/            bundled pip wheels for offline installation
run.py                    required evaluator entrypoint
requirements-backend.txt  backend runtime dependencies
requirements-frontend.txt visualization dependencies
```
