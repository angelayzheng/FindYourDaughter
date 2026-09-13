# Status of labeling the 25 scans

**The 25 scans are not yet a correctly labeled comparison dataset.** No expert
annotations or qualified reviewer are currently available. Cases 19-23 contain
the existing 19 draft instances; the other 20 cases have no reference labels.
Anatomical correctness and completeness remain unverified for every case.

The [per-case readiness record](annotation_status_20260913.md) lists all 25 scans,
their existing-label coverage, working spacing, and outstanding status.

The new annotation workspace prepares all scans for labeling. It does not
manufacture ground truth from the models being compared, create empty-negative
labels for unannotated scans, or change existing models and evaluation results.
The five-case draft set remains usable for its previously documented limited
agreement measurements.

## Open the local workspace

Open `annotation_workspaces/20260913_all25_v2/index.html` in a local browser.
Each case contains:

- `ct_working.nii.gz` and `aorta_working.nii.gz`: a checked, matching pair for
  annotation. The original source files remain unchanged. Case 24 uses the
  existing loader's orthogonal resampling; the workspace records this explicitly.
- `review.html`: a self-contained, offline three-plane CT viewer, with all
  slices through the supplied parent region and a 25 mm surrounding margin.
  Window and level can be changed. Hints start hidden to support an initial
  review that is independent of model predictions.
- `annotations.json`: annotation status, input hashes, grid transforms, review
  fields, and separate **unreviewed proposal cards**. `daughters` is `null`
  (unknown), and `comparison_ready` is `false`. There is no daughter-reference
  NIfTI for the unannotated cases.
- `model_hints/`: optional Baseline, Contact, default Refined, and configured
  Refined predictions with full provenance. Proposal counts are not branch
  counts. Different cards can represent the same origin; exact duplicate
  geometry is consolidated only as a viewing convenience.

`existing_draft/` is a preserved copy of the original five-case package. Input
hashes are checked before associating each of those drafts with a source scan.
Its review status has not been upgraded. The incomplete first preparation pass
under `20260913_all25/` is retained; use the completed `v2` directory.

The HTML preview stores rounded, bounded signed-16-bit CT values for display;
the working NIfTI retains the original loaded voxel values, or the loader's
resampled values for case 24. Every saved working volume is reopened and checked
for identical voxel values. CT/mask grids must match. NIfTI header rounding is
measured at all grid corners and limited to 0.001 mm; all viewer coordinates
are calculated from the reopened working grid. This avoids mixing original
double-precision coordinates with the NIfTI-1 header's rounded transform.

## Record review observations

Scroll through all three planes and the full supplied aortic extent. Clicking a
plane moves the crosshair in that plane. A manual guide can be started with
**New guide**, then extended using **Add crosshair point**. Guide points are
saved as voxel XYZ and physical SimpleITK LPS millimetres. A displayed guide
length is not proof of continuous daughter lumen, origin size, or eligibility.

Reveal model hints only when needed. Selecting a proposal shows its proposed
ostium and seed near the displayed slices. Record `retain_for_review`,
`exclude_from_draft`, or `unresolved` with evidence; none of these is a certified
anatomical decision. No reviewer name, date, decision, or completeness sign-off
is filled automatically.

**Download review JSON** saves observations as a new file. **Resume review**
loads a matching case/input version. These records remain unreviewed and cannot
be exported by this viewer as scoring references. The viewer does not paint
voxel masks or calculate validated origin diameters or seed radii; final mask
editing and anatomical adjudication remain outstanding.

Resume also checks the working grid, model-hint configuration, proposal
identity, and agreement between voxel and physical guide coordinates. It
refuses a record from a different grid or with inconsistent point coordinates.

## Requirements before reference comparison

For each retained origin, establish direct connection to the parent in
consecutive orthogonal CT slices, distinguish separate ostia from common trunks,
and exclude crop faces and terminal iliac division. Trace visible lumen for at
least 5 mm and stop at 10 mm or the first downstream bifurcation. Record early
forks and unresolved origins explicitly. Check origin diameter separately from
seed radius; the **2 mm cutoff is provisional**, inherited from the old draft
package. The challenge PDF defers the final minimum size to the final dataset.

The entire parent circumference must also be reviewed for origins missed by
every model. Correct masks, physical landmarks, directions, instance identities,
and eligibility need evidence and a review record. Structural checks and model
consensus cannot establish anatomical correctness. A partly labeled case is
not a complete negative inventory for precision/F1 measurement.

The scorer rejects workspace cases marked `comparison_ready: false` before
loading labels or running any detector. Setting this field to `true` alone
does not create valid annotations or establish accuracy; the existing full
reference-schema and geometry checks still apply. Legacy five-case draft
evaluation behavior is unchanged.

## Reproduce preparation

Use a new output directory:

```powershell
python scripts/prepare_annotation_workspace.py --dataset dataset --existing-draft eval_set --hint-config configs/tuned_20260913/refined.json --output-dir annotation_workspaces/new_review
```

`--without-model-hints` prepares image review without running any detector.
Existing output directories are rejected. Preparation is CPU-only and needs
no network or additional dependencies. Generated scan data and hints are
ignored by version control; the preparation tools and status documentation are
retained in the repository.
