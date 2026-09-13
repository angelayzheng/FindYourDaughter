# Annotation readiness record - 2026-09-13

**No case is certified ground truth.** The workspace is complete; reference labeling is not.
The original 19 draft labels on five cases are preserved. Twenty cases still have no reference labels.

Open the local `annotation_workspaces/20260913_all25_v2/index.html` and see [the workflow](annotation_workflow.md).
All 50 source files and all copied existing-draft files were verified unchanged. The saved CT/mask pairs reopen with identical voxel values.
All unreviewed cases are rejected by the scorer. Model hints were not promoted to references.

| Subject | Working spacing XYZ (mm) | Existing draft labels | Model proposal cards (not branches) | Annotation status |
| --- | --- | ---: | ---: | --- |
| subject001 | 0.782 x 0.782 x 0.800 | Unknown | 13 | Unlabeled |
| subject002 | 0.686 x 0.686 x 0.800 | Unknown | 15 | Unlabeled |
| subject003 | 0.732 x 0.732 x 0.800 | Unknown | 22 | Unlabeled |
| subject004 | 0.625 x 0.625 x 0.800 | Unknown | 23 | Unlabeled |
| subject005 | 0.900 x 0.900 x 0.800 | Unknown | 24 | Unlabeled |
| subject006 | 0.774 x 0.774 x 0.800 | Unknown | 20 | Unlabeled |
| subject007 | 0.862 x 0.862 x 0.800 | Unknown | 9 | Unlabeled |
| subject008 | 0.647 x 0.647 x 0.800 | Unknown | 23 | Unlabeled |
| subject009 | 0.850 x 0.850 x 0.800 | Unknown | 23 | Unlabeled |
| subject010 | 0.961 x 0.961 x 0.800 | Unknown | 17 | Unlabeled |
| subject011 | 0.839 x 0.839 x 0.800 | Unknown | 26 | Unlabeled |
| subject012 | 0.782 x 0.782 x 0.800 | Unknown | 21 | Unlabeled |
| subject013 | 0.858 x 0.858 x 0.800 | Unknown | 31 | Unlabeled |
| subject014 | 0.702 x 0.702 x 0.800 | Unknown | 19 | Unlabeled |
| subject015 | 0.726 x 0.726 x 0.800 | Unknown | 30 | Unlabeled |
| subject016 | 1.500 x 1.500 x 1.500 | Unknown | 7 | Unlabeled |
| subject017 | 1.500 x 1.500 x 1.500 | Unknown | 3 | Unlabeled |
| subject018 | 1.500 x 1.500 x 1.500 | Unknown | 19 | Unlabeled |
| subject019 | 1.500 x 1.500 x 1.500 | 3 | 4 | Existing draft; unverified |
| subject020 | 1.500 x 1.500 x 1.500 | 4 | 4 | Existing draft; unverified |
| subject021 | 1.500 x 1.500 x 1.500 | 3 | 19 | Existing draft; unverified |
| subject022 | 1.500 x 1.500 x 1.500 | 6 | 20 | Existing draft; unverified |
| subject023 | 1.500 x 1.500 x 1.500 | 3 | 4 | Existing draft; unverified |
| subject024 | 1.500 x 1.500 x 1.500 | Unknown | 12 | Unlabeled; working grid resampled |
| subject025 | 1.500 x 1.500 x 1.500 | Unknown | 27 | Unlabeled |

There are 435 proposal cards from four model configurations. Different cards can be observations of the same origin, and all models can miss an origin.
Neither these counts nor agreement between models measures true branch prevalence.

Sample exported CT/parent views were inspected for every case. Cases 16, 18 and 24 show marked voxel-scale intensity variation in those views; the effect on each proposed branch remains unreviewed.
This was an export-alignment check, not a full-volume sweep or daughter-label adjudication. Case 24 uses a recorded resampled working grid.

146 automated tests passed. Viewer rendering/navigation and coordinate/resume logic were exercised in a simulated DOM for all 25 pages.
No controlled browser was available, so actual browser appearance, clicks and downloads were not visually verified.

The remaining work is anatomical labeling and review: establish every eligible direct origin, trace its lumen, measure landmarks/radii, resolve forks and duplicates, edit the masks, and audit the whole parent for omissions.
