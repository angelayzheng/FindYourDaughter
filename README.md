# Find Your Daughter

Branchseed Challenge submission. The current backend validates input geometry and writes an empty `daughters` list; branch detection is not implemented yet.

The bundled offline wheels target CPython 3.14 on Windows x64. From the repository root, install dependencies:

```text
python scripts/install_offline.py
```

Run a case:

```text
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

After setup, open the native 3D viewer (it automatically loads a single neighboring mask):

```text
python scripts/view_nifti_3d.py --image dataset/subject016/orig16.nii
```

See [3D viewer controls, options, and screenshots](docs/development.md#native-3d-volume-viewer) or the [development guide](docs/development.md) for details.
