# UTWAT Battle of the Schools Hackathon

## Inspect CT dimensions

Install the NIfTI reader:

```bash
python3 -m pip install -r requirements.txt
```

Inspect every `.nii` or `.nii.gz` file recursively. The script reads image
headers only, so it does not load the full CT volumes into memory:

```bash
python3 scripts/inspect_nifti_dataset.py dataset
```

The report includes voxel shape, voxel spacing, approximate physical size in
millimetres, data type, subject, and whether the file is an `orig` image or a
`mask`. Export machine-readable records with:

```bash
python3 scripts/inspect_nifti_dataset.py dataset --output nifti_dimensions.csv
python3 scripts/inspect_nifti_dataset.py dataset --output nifti_dimensions.json
```

## Branch-seed runner

The challenge specification requires one JSON prediction file per CT/mask pair.
Install the runtime dependencies and run the current valid baseline with:

```bash
python3 -m pip install -r requirements.txt
python3 run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output prediction.json
```

The baseline validates that the volumes share the same 3-D physical grid and
writes the required schema. Daughter detection is the next algorithm layer;
until it is implemented, the output contains an empty `daughters` list.
