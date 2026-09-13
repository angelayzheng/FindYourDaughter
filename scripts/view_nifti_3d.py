"""Open one NIfTI case in a native VTK desktop window (no web server)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import ScanCase, VolumeViewOptions
from backend.detectors import DETECTOR_NAMES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="CT .nii or .nii.gz")
    source.add_argument("--subject", type=Path, help="Subject folder; discover CT and mask automatically")
    parser.add_argument("--detect", action="store_true",
                        help="Run the experimental detector and overlay branches on its working grid")
    parser.add_argument("--detector", choices=DETECTOR_NAMES,
                        help="Algorithm to overlay (requires --detect; default: refined)")
    parser.add_argument("--config", type=Path, help="Saved detector configuration (requires --detect)")
    parser.add_argument("--branch", type=int,
                        help="Initially focus this candidate number (1-based; requires --detect)")
    parser.add_argument(
        "--mask", "--aorta-mask", dest="mask",
        type=Path,
        help="Optional matching mask; otherwise detect one neighboring mask*.nii",
    )
    parser.add_argument(
        "--no-mask", action="store_true", help="Disable neighboring-mask detection"
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Frame of a 4-D image (default: 0)"
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=192,
        help="Maximum sampled CT dimension, 16-512 (default: 192)",
    )
    parser.add_argument(
        "--window",
        type=float,
        help="Intensity window width (default: 400, or 600 with --detect)",
    )
    parser.add_argument(
        "--level", type=float, help="Intensity window center (default: 40, or 200 with --detect)"
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        help="Hide volume and slice pixels below this intensity",
    )
    parser.add_argument("--denoise", action="store_true", help="Suppress voxels without locally similar-intensity neighbors")
    parser.add_argument("--denoise-tolerance", type=float, default=40.0, help="Denoiser intensity tolerance (default: 40)")
    parser.add_argument("--denoise-neighbors", type=int, default=2, help="Required similar neighbors, 1-26 (default: 2)")
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.12,
        help="Maximum volume opacity, 0-0.5 (default: 0.12)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save an initial PNG and use this path for the S key",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Save --screenshot and exit without opening the viewer",
    )
    args = parser.parse_args(argv)
    if args.subject is not None:
        from backend.cli import discover_subject_inputs

        try:
            args.image, discovered_mask = discover_subject_inputs(args.subject)
        except ValueError as error:
            parser.error(str(error))
        if args.mask is None:
            args.mask = discovered_mask
        if args.no_mask:
            parser.error("--subject cannot be combined with --no-mask")
    if args.mask is not None and args.no_mask:
        parser.error("--mask and --no-mask cannot be used together")
    if args.offscreen and args.screenshot is None:
        parser.error("--offscreen requires --screenshot")
    if args.branch is not None and (not args.detect or args.branch < 1):
        parser.error("--branch requires --detect and a positive candidate number")
    if args.detector is not None and not args.detect:
        parser.error("--detector requires --detect")
    if args.config is not None and not args.detect:
        parser.error("--config requires --detect")
    if args.detect and (args.no_mask or args.frame != 0):
        parser.error("--detect requires an aorta mask and a 3-D scan (--frame 0)")
    try:
        if args.detect:
            from backend.configuration import configuration, load_configuration
            config = load_configuration(args.config, detector=args.detector) if args.config else configuration(args.detector or "refined")
        options = VolumeViewOptions(
            frame=args.frame,
            max_dimension=args.max_dimension,
            window=args.window if args.window is not None else (600 if args.detect else 400),
            level=args.level if args.level is not None else (200 if args.detect else 40),
            min_intensity=args.min_intensity,
            denoise=args.denoise,
            denoise_tolerance=args.denoise_tolerance,
            denoise_min_neighbors=args.denoise_neighbors,
            opacity=args.opacity,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    mask_path = args.mask
    if mask_path is None and not args.no_mask and "mask" not in args.image.name.lower():
        candidates = sorted(
            p
            for p in args.image.parent.iterdir()
            if p.is_file()
            and "mask" in p.name.lower()
            and p.name.lower().endswith((".nii", ".nii.gz"))
        )
        if len(candidates) > 1:
            parser.error(
                "Several neighboring masks found; choose --mask explicitly or use --no-mask"
            )
        mask_path = candidates[0] if candidates else None
    from desktop.volume_viewer import VolumeViewer

    detection = None
    if args.detect:
        if mask_path is None:
            parser.error("--detect requires an aorta mask; provide --mask")
        from backend.inputs import load_case
        from backend.detectors import detect
        from desktop.detection_overlay import scan_case_from_backend

        print(f"Loading and detecting candidate arteries ({config['detector']})...", flush=True)
        working_case = load_case(args.image, mask_path)
        detection = detect(working_case.image, working_case.aorta_mask,
                           detector=config["detector"], parameters=config["parameters"])
        case = scan_case_from_backend(working_case)
        del working_case
        print(f"{len(detection.branches)} experimental candidates; Left / Right to browse, J to focus, D to toggle", flush=True)
        if args.branch is not None and args.branch > len(detection.branches):
            parser.error(f"--branch {args.branch} exceeds the {len(detection.branches)} detected candidates")
    else:
        case = ScanCase.from_nifti(args.image, mask_path)
    viewer = VolumeViewer(case, options, detection=detection)
    if args.branch is not None:
        viewer.detection_overlay.select(args.branch - 1)
    viewer.show(screenshot=args.screenshot, offscreen=args.offscreen)


if __name__ == "__main__":
    main()
