# AGENTS.md

## Project purpose

This repository is a submission for the Branchseed Challenge. The authoritative
requirements are in `Branchseed challenge.pdf`.

The system receives a CT NIfTI volume and a binary mask containing only the
parent abdominal aorta. It must eventually detect every eligible artery that
arises directly from the aorta and report each daughter as a separate instance.

## Non-negotiable evaluator interface

The project must continue to support this command from the repository root:

```text
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Do not rename or relocate `run.py`. Keep the evaluator-facing backend independent
of Streamlit and other optional user-interface code.

The output must be valid JSON with this envelope:

```json
{
  "case_id": "subject001",
  "parent": {"instance_id": "aorta"},
  "daughters": []
}
```

Each future daughter object must contain a unique `branch_NNN` instance ID,
`parent_instance_id` set to `aorta`, physical-space ostium and seed coordinates,
a radius in millimetres, and a unit direction vector pointing into the daughter.

## Challenge constraints

- Judging is offline: runtime code must never require network access.
- Execution is CPU-only on a standard laptop with four CPU cores and 8 GB RAM.
- Target average runtime is at most 60 seconds per case unless organizers revise it.
- Inputs are 3-D NIfTI images whose image and mask grids and physical geometry match.
- Preserve the SimpleITK physical-coordinate system.
- Convert voxel indices with `SimpleITK.TransformIndexToPhysicalPoint`; never
  report voxel indices as physical coordinates.
- Do not segment the parent aorta, assign anatomical vessel names, infer branches
  outside the scan, or reconstruct the complete distal vascular tree.
- Cropped superior and inferior mask faces are not branch origins.
- A common trunk is one direct daughter; a branch arising from another daughter
  is not a direct aortic daughter.
- Keep separate ostia as separate instances and avoid duplicate detections.
- The terminal iliac division is outside the core task.

## Repository layout

- `backend/`: evaluator-facing processing and JSON output code.
- `frontend/`: optional Streamlit visualization only.
- `scripts/`: development utilities and offline dependency tooling.
- `scripts/visualize_nifti.py`: quick-look NIfTI preview utility.
- `vendor/wheels/`: wheels bundled for offline pip installation.
- `tests/`: automated tests.
- `docs/`: detailed setup, development, data-inspection, and visualization guides.
- `run.py`: required evaluator entrypoint.

Keep detection logic in `backend/`. The frontend may call backend code but backend
code must not import from `frontend/`.

## Dependencies and offline installation

Backend dependencies belong in `requirements-backend.txt`. Visualization-only
dependencies belong in `requirements-frontend.txt`. Pin direct dependencies to
specific versions.

On an internet-connected preparation machine matching the judge's operating
system, architecture, and Python version, rebuild the wheelhouse with:

```text
python scripts/download_wheels.py
```

The offline setup command is:

```text
python scripts/install_offline.py
```

The installer must retain `--no-index`, validate the wheelhouse platform, and
verify wheel hashes. Never assume binary wheels are portable between Python
versions or operating systems. Commit the populated wheelhouse in the final
submission archive.

## Development conventions

- Follow `CONTRIBUTING.md` for branch names, Conventional Commits, and pull
  request expectations.
- Support Windows paths and avoid shell-specific runtime assumptions.
- Use `pathlib.Path` for filesystem paths.
- Keep imports side-effect free except in the Streamlit entrypoint.
- Keep coordinates, spacing, distances, and radii explicitly identified as
  either voxel-space or physical millimetres.
- Prefer deterministic algorithms and stable output ordering.
- Avoid GPU-only libraries, heavyweight models, and unnecessary full-volume
  copies because of the evaluation limits.
- Do not add case-specific constants or manual point placement.
- Keep generated predictions, previews, virtual environments, caches, datasets,
  and source archives out of version control unless they are explicit submission
  artifacts.

When suggesting or creating a commit, use this structure:

```text
<type>: <imperative summary>
```

When suggesting or creating a branch, use `<type>/<short-kebab-case-description>`.

## Documentation

- Keep `README.md` extremely short and focused on judging: the offline setup
  command, the required evaluator command, and a brief statement of the current
  detection status. Link to `docs/` for everything else.
- Put detailed usage, architecture, data-inspection, visualization, and developer
  instructions in `docs/`. Keep `CONTRIBUTING.md` for contribution conventions.
- Update the relevant documentation in the same change whenever behavior,
  commands, dependencies, supported platforms, output fields, or limitations
  change. Remove or correct instructions that no longer match the code.
- Keep evaluator commands and current implementation status accurate. Do not
  document planned branch detection as working functionality.
- Review documentation during verification, including changes that appear to
  affect only internal code. Documentation-only edits need `git diff --check`;
  run executable examples when they change.

## Verification

After backend or infrastructure changes, run:

```text
python -m unittest discover -s tests -v
python run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --output prediction.json
python -m pip check
git diff --check
```

After frontend changes, also verify that this starts without network access:

```text
python -m streamlit run frontend/app.py
```

Do not claim branch detection accuracy while `backend/pipeline.py` still returns
the intentionally empty scaffold. When detection is implemented, add tests for
physical-coordinate conversion, direction normalization, instance uniqueness,
empty cases, cropped mask ends, common trunks, and geometry mismatches.
