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

See [development and visualization guidance](docs/development.md) for other commands and details.
