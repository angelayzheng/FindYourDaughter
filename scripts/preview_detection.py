"""Export experimental branch candidates and CT overlays for manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import SimpleITK as sitk

from backend.detectors import DETECTOR_NAMES, detect
from backend.inputs import load_case
from backend.pipeline import _case_id


def export_overlays(case, result, output: Path) -> list[Path]:
    """Show every candidate in three local projections on the working grid."""
    import matplotlib.pyplot as plt

    case_id = _case_id(case.image_path)
    image = sitk.GetArrayViewFromImage(case.image)
    mask = sitk.GetArrayViewFromImage(case.aorta_mask)
    spacing = np.array(case.image.GetSpacing())[::-1]
    paths = []
    branches = result.branches
    for page in range(max(1, (len(branches) + 5) // 6)):
        batch = branches[page * 6:(page + 1) * 6]
        fig, axes = plt.subplots(max(1, len(batch)), 3, figsize=(13, max(4, 3.5 * len(batch))), squeeze=False,
                                 layout="constrained")
        if not batch:
            for ax in axes.flat:
                ax.axis("off")
            axes[0, 1].text(.5, .5, "No candidates passed the current filters", ha="center", va="center")
        for row, branch in enumerate(batch):
            physical = np.array(branch.centerline_xyz_mm)
            voxels = np.array([case.image.TransformPhysicalPointToContinuousIndex(point.tolist())[::-1]
                               for point in physical])
            root = voxels[0]
            seed = np.array(case.image.TransformPhysicalPointToContinuousIndex(branch.seed_xyz_mm)[::-1])
            lo = np.maximum(0, np.floor(voxels.min(axis=0) - 7 / spacing).astype(int))
            hi = np.minimum(image.shape, np.ceil(voxels.max(axis=0) + 7 / spacing).astype(int) + 1)
            slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            local_ct, local_mask = image[slices], mask[slices]
            for column, axis in enumerate((0, 1, 2)):
                ax = axes[row, column]
                plane_axes = [dim for dim in range(3) if dim != axis]
                vertical, horizontal = plane_axes
                # A narrow slab around the root avoids obscuring the opening.
                center = int(round(root[axis] - lo[axis]))
                half = max(1, int(np.ceil(2 / spacing[axis])))
                slab = [slice(None)] * 3
                slab[axis] = slice(max(0, center - half), min(local_ct.shape[axis], center + half + 1))
                ct_view = local_ct[tuple(slab)].max(axis=axis)
                mask_view = local_mask[tuple(slab)].max(axis=axis)
                extent = ((lo[horizontal]-.5)*spacing[horizontal], (hi[horizontal]-.5)*spacing[horizontal],
                          (hi[vertical]-.5)*spacing[vertical], (lo[vertical]-.5)*spacing[vertical])
                ax.imshow(ct_view, cmap="gray", vmin=-100,
                          vmax=max(350, result.diagnostics.get("blood_intensity", 300) + 120), extent=extent)
                if np.any(mask_view) and not np.all(mask_view):
                    ax.contour(np.arange(lo[horizontal], hi[horizontal])*spacing[horizontal],
                               np.arange(lo[vertical], hi[vertical])*spacing[vertical],
                               mask_view, levels=[.5], colors=["#ff4242"], linewidths=1)
                ax.plot(voxels[:, horizontal]*spacing[horizontal], voxels[:, vertical]*spacing[vertical],
                        color="#35e0ff", linewidth=1.4)
                ax.scatter(root[horizontal]*spacing[horizontal], root[vertical]*spacing[vertical],
                           c="yellow", s=28, edgecolors="black", linewidths=.4, zorder=5)
                ax.annotate("", xy=(seed[horizontal]*spacing[horizontal], seed[vertical]*spacing[vertical]),
                            xytext=(root[horizontal]*spacing[horizontal], root[vertical]*spacing[vertical]),
                            arrowprops={"arrowstyle": "->", "color": "#35e0ff", "lw": 1.5})
                ax.set_title(f"branch_{page*6+row+1:03d} | {('native XY', 'native XZ', 'native YZ')[column]} | r={branch.radius_mm:.1f} mm")
                ax.set_xlabel("native grid distance (mm)")
                ax.set_aspect("equal")
        fig.suptitle(f"{case_id} | {result.diagnostics.get('detector', 'baseline')} | EXPERIMENTAL candidates, not reference annotations\n"
                     "red: parent mask | yellow: proposed ostium | cyan: projected path and direction to 5 mm seed\n"
                     "CT/mask show a narrow slab at the ostium; projected paths can extend outside this slab",
                     fontsize=12)
        path = output / f"{case_id}_candidates_{page+1:02d}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    # Remove stale pages from an earlier run that produced more candidates.
    prefix = f"{case_id}_candidates_"
    for stale in output.iterdir():
        if (stale.is_file() and not stale.is_symlink() and stale.suffix == ".png"
                and stale.stem.startswith(prefix) and stale.stem[len(prefix):].isdigit()
                and stale not in paths):
            stale.unlink()
    return paths


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--aorta-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: nifti_previews/detection, with a subfolder for alternative detectors")
    parser.add_argument("--detector", choices=DETECTOR_NAMES)
    parser.add_argument("--config", type=Path, help="Saved detector configuration")
    args = parser.parse_args()
    from backend.configuration import configuration, load_configuration
    try:
        config = load_configuration(args.config, detector=args.detector) if args.config else configuration(args.detector or "baseline")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.output_dir is None:
        args.output_dir = Path("nifti_previews/detection")
        if config["detector"] != "baseline":
            args.output_dir /= config["detector"]
    case = load_case(args.image, args.aorta_mask)
    result = detect(case.image, case.aorta_mask, detector=config["detector"], parameters=config["parameters"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_id = _case_id(args.image)
    prediction = {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": result.daughters()}
    (args.output_dir / f"{case_id}_prediction.json").write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
    details = {"diagnostics": result.diagnostics,
               "centerlines_xyz_mm": [branch.centerline_xyz_mm for branch in result.branches],
               "radius_methods": [branch.radius_method for branch in result.branches]}
    (args.output_dir / f"{case_id}_diagnostics.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    export_overlays(case, result, args.output_dir)
    print(f"{case_id}: {len(result.branches)} experimental candidates; outputs in {args.output_dir}")
    # Per-contact paths can be large; keep them in the diagnostic file.
    verbose = {"contacts", "candidates", "source_contact_records", "deferred_common_trunks"}
    print(json.dumps({key: value for key, value in result.diagnostics.items() if key not in verbose}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
