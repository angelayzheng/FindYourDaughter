"""Open one NIfTI case in a native VTK desktop window (no web server)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import ScanCase, VolumeViewOptions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="CT .nii or .nii.gz")
    parser.add_argument("--mask", type=Path, help="Optional matching mask; otherwise detect one neighboring mask*.nii")
    parser.add_argument("--no-mask", action="store_true", help="Disable neighboring-mask detection")
    parser.add_argument("--frame", type=int, default=0, help="Frame of a 4-D image (default: 0)")
    parser.add_argument("--max-dimension", type=int, default=192, help="Maximum sampled CT dimension, 16-512 (default: 192)")
    parser.add_argument("--window", type=float, default=400, help="Intensity window width (default: 400)")
    parser.add_argument("--level", type=float, default=40, help="Intensity window center (default: 40)")
    parser.add_argument("--min-intensity", type=float, help="Hide volume and slice pixels below this intensity")
    parser.add_argument("--opacity", type=float, default=0.12, help="Maximum volume opacity, 0-0.5 (default: 0.12)")
    parser.add_argument("--screenshot", type=Path, help="Save an initial PNG and use this path for the S key")
    parser.add_argument("--offscreen", action="store_true", help="Save --screenshot and exit without opening the viewer")
    args = parser.parse_args(argv)
    if args.mask is not None and args.no_mask:
        parser.error("--mask and --no-mask cannot be used together")
    if args.offscreen and args.screenshot is None:
        parser.error("--offscreen requires --screenshot")
    try:
        options = VolumeViewOptions(frame=args.frame, max_dimension=args.max_dimension,
                                    window=args.window, level=args.level,
                                    min_intensity=args.min_intensity, opacity=args.opacity)
    except ValueError as error:
        parser.error(str(error))
    mask_path = args.mask
    if mask_path is None and not args.no_mask and "mask" not in args.image.name.lower():
        candidates = sorted(p for p in args.image.parent.iterdir() if p.is_file()
                            and "mask" in p.name.lower() and p.name.lower().endswith((".nii", ".nii.gz")))
        if len(candidates) > 1:
            parser.error("Several neighboring masks found; choose --mask explicitly or use --no-mask")
        mask_path = candidates[0] if candidates else None
    from desktop.volume_viewer import VolumeViewer

    case = ScanCase.from_nifti(args.image, mask_path)
    viewer = VolumeViewer(case, options)
    viewer.show(screenshot=args.screenshot, offscreen=args.offscreen)


if __name__ == "__main__":
    main()
