# Find Your Daughter

Branchseed Challenge submission. The CPU backend detects direct daughter
arteries from a CT NIfTI volume and a parent-only aorta mask. Detection is
offline; expert-validated accuracy is not yet established.

Install the bundled CPython 3.14 / Windows x64 dependencies offline:

```text
python scripts/install_offline.py
```

Run from the repository root with the required evaluator interface:

```text
python run.py --image <image_path>.nii.gz --aorta-mask <aorta_mask_path>.nii.gz --output prediction.json
```

The output is JSON containing `case_id`, the parent `aorta`, and detected
daughter instances with physical-space coordinates, radius, and direction.

See [docs/](docs/) for development and evaluation details.
