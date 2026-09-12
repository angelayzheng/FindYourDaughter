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

## Verification

Run the backend smoke tests with:

```powershell
python -m unittest discover -s tests -v
```

## Layout

```text
backend/                  evaluator-facing Python package
eda/                      dataset inspection utilities
frontend/app.py           optional Streamlit entrypoint
scripts/                  visualization and offline dependency tooling
vendor/wheels/            bundled pip wheels for offline installation
run.py                    required evaluator entrypoint
requirements-backend.txt  backend runtime dependencies
requirements-frontend.txt visualization dependencies
```
