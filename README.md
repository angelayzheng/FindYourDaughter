# Find Your Daughter

Branchseed Challenge submission. The CPU backend proposes direct daughter
arteries from CT and a supplied parent-aorta mask. The original baseline remains
the default; [contact](docs/contact_detection.md), [fusion](docs/fusion_detection.md),
and [refined](docs/refined_detection.md)
alternatives are available. [Draft-reference evaluation](docs/evaluation.md) is supported;
expert-validated detection accuracy remains unmeasured.

The bundled offline wheels target CPython 3.14 on Windows x64:

```text
python scripts/install_offline.py
```

Run from the repository root:

```text
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

See [detection and native 3D review](docs/artery_detection.md),
[setup, data inspection, and visualization](docs/development.md), or
[contributing](CONTRIBUTING.md) for details.
