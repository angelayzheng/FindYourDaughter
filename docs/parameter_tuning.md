# Reproducible detector parameter tuning

Original defaults remain available and reproduce the existing predictions.
Tuning creates named configurations, not replacements for the existing models.
The evaluator JSON envelope and daughter fields are unchanged; parameter
provenance belongs in configuration files and diagnostic/evaluation reports.

The completed [114-trial experiment](experiments/tuning_20260913.md) and
[evaluation comparison](evaluation.md#parameter-tuning-results-2026-09-13)
record the selected settings and their tradeoffs. Reusable full configurations
are stored under `configs/tuned_20260913/`; `refined.json` is the selected
setting, while `refined_draft_f1.json` is the higher draft-F1 alternative that
failed the synthetic development guard.

## Configure a detector

A configuration selects one detector and contains named parameter groups. Only
changed fields need to be supplied; the loader expands every omitted field to
its original default. For example:

```json
{
  "detector": "refined",
  "parameters": {
    "contact": {"intensity_fraction": 0.30, "min_vesselness": 0.15},
    "refined": {"minimum_natural_sections": 1}
  }
}
```

`baseline` accepts its `baseline` group; `contact` accepts its `contact` group.
`fusion` accepts `baseline`, `contact`, and `fusion`; `refined` accepts all four.
Unknown groups/fields, nonfinite values, wrong types, and invalid ranges fail
before inference. An explicitly supplied `--detector` must agree with the file.
The detector in the file is used when `--detector` is omitted. Without either
flag, `run.py` retains Baseline and the evaluation runner retains all detectors.

The groups expose `DetectionOptions`, `ContactOptions`, `FusionOptions`, and
`RefinedOptions`. Distances ending in `_mm` are physical millimetres. Refined's
repair limits are multiples of a physical voxel diagonal, not fixed voxel
indices. Seed distance, coordinate conventions, and the evaluator contract
are not tuning parameters.

For Contact, `intensity_fraction` sets the lower threshold to
`max(40, background + fraction * (blood - background))` HU, using measurements
from the input CT. Lower values admit dimmer lumen but also change contact and
skeleton topology. They need not increase recall. `min_vesselness` requires
stronger tube evidence when increased; it is an evidence score, not a calibrated
probability. Each detector and each source group retains its own settings.

| Group | Available parameters |
| --- | --- |
| `baseline` | `margin_mm`, `intensity_fraction`, `min_radius_mm`, `max_radius_mm`, `min_vesselness` |
| `contact` | `margin_mm`, `growth_mm`, `intensity_fraction`, `support_band_mm`, `tube_seed`, `tube_reach_mm`, `min_vesselness`, `min_radius_mm`, `max_radius_mm` |
| `fusion` | `duplicate_ostium_mm`, `duplicate_path_mm`, `minimum_support_fraction`, `max_radius_cv`, `max_expansion_ratio` |
| `refined` | `max_initial_chord_voxel_diagonals`, `max_origin_shift_voxel_diagonals`, `minimum_closed_sections`, `minimum_natural_sections` |

Refined defaults require at least one closed section and zero naturally closed
sections, reproducing its previous acceptance rule. A naturally closed section
does not need the parent mask removed to obtain closure. Increasing either
section count adds an explicit gate and can reject real short/adjacent branches;
the search must measure that tradeoff. Counts are integers and apply only to
the available measured sections at 4, 5, and 6 mm.

Python callers can use `backend.detectors.detect(..., detector="refined",
parameters={"contact": {"intensity_fraction": 0.30}})`. The registry records the
fully resolved configuration in `result.diagnostics["configuration"]`.

## Run a preserved search

```powershell
python scripts/tune_detectors.py sweep --plan configs/tuning_grid.json --dataset eval_set --synthetic-development tmp/contact-validation-e3ece44b43214023aabcf8622c276472/synthetic
```

The default destination is a unique UTC timestamp under `nifti_previews/tuning/`.
An explicit `--output-dir` must not exist; rerunning into an existing experiment
is rejected. Old evaluation directories and historical tables are not replaced.
The supplied grid contains **114 distinct configurations**: 12 Baseline,
30 Contact, 24 Fusion, and 48 Refined. It always includes each detector's
unchanged default. Edit a copy of the plan to explore a different grid in a
new experiment. Cartesian-product combinations are validated before any run.

The development tubes in the local command above use seed 941, ten cases,
size 128 x 128 x 192, and spacing 0.8 x 0.8 x 1.0 mm. If that ignored local
directory is unavailable, generate the cohort at a new path and pass it instead:

```powershell
python scripts/generate_synthetic_cases.py --output tmp/tuning-development --cases 10 --seed 941 --size 128 128 192
```

Each experiment stores:

- `plan.json` and `configurations.json`: the search specification and every
  effective configuration, including defaults.
- `trials/<detector>-<configuration-hash>/config.json`, `report.json`, and
  `summary.md`, plus each case's prediction and diagnostics. Reports include
  input/code hashes, versions, timing, failures, matching details, and errors.
- `index.json` and `summary.md`: the cumulative trial record and comparison.
  The index is updated during this run; completed trial directories are retained.
- `selected/<detector>.json`: the optional selected settings after the
  synthetic development check. Existing defaults are never rewritten.

Detection runs serially on CPU without network access. Input hashes and backend
hashes must stay constant through a search. Interrupted/failed runs retain
completed files; there is currently no resume mode. Failed or incomplete trials
cannot win, and an incomplete experiment cannot proceed to holdout validation.

## What "best" means

The declared ranking is **highest micro-averaged draft F1**, then highest recall,
then fewer settings changed from default, then deterministic configuration ID.
Runtime is recorded but is not a noisy tie-breaker. Matching tolerance remains
**3 mm** for every trial; changing scoring tolerance is not a model improvement.

After the draft sweep, the best three configurations per detector and its
default are evaluated on synthetic development data. A candidate is eligible
only if its synthetic F1 and recall are at least its detector default's values.
The runner selects the best draft-ranked eligible member of that shortlist.
The raw draft leader remains recorded even when the guard disqualifies it.
This is the best observed setting under this bounded search and selection rule,
not a global optimum, and not proof of anatomical accuracy.

All five draft cases informed this search. They are not held out, are not
expert-validated, and may omit eligible branches. Optimizing agreement with
them can reward suppressing real unannotated origins. Synthetic checks help
detect regressions but do not remove that limitation.

## Validate frozen selections

Generate a fresh cohort after selection, then use `validate` with its path:

```powershell
python scripts/generate_synthetic_cases.py --output tmp/tuning-holdout --cases 16 --seed 19337 --size 128 128 192
python scripts/tune_detectors.py validate --run-dir nifti_previews/tuning/20260913_f1_grid --dataset tmp/tuning-holdout
```

Validation runs each default and its selected configuration once, deduplicating
identical settings. It records results under the experiment's `holdout/`
directory without changing the selection. It refuses repeated validation in
that directory, changed backend code, and reused development CT/truth files.
It cannot prove that arbitrary user-supplied data were never inspected before;
keep the experimental split honest. The reported fresh cohort was not used to
choose parameter values.

## Reuse a configuration

The recorded experiment's selected Refined configuration can
be used in the evaluator, scorer, or native viewer:

```powershell
python run.py --image dataset/subject001/orig1.nii --aorta-mask dataset/subject001/mask1.nii --config configs/tuned_20260913/refined.json --output nifti_previews/tuned_prediction.json
python scripts/evaluate_detectors.py --config configs/tuned_20260913/refined.json --output-dir nifti_previews/tuned_refined_check
python scripts/view_nifti_3d.py --image dataset/subject001/orig1.nii --detect --config configs/tuned_20260913/refined.json
```

Static previews and `scripts/evaluate_synthetic_detection.py` also accept
`--config`. Supply a fresh preview/evaluation output path to preserve previous
exports; only the tuning runner enforces a new experiment directory.
