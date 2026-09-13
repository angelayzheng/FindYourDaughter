"""Benchmark per-case artery detection time for local NIfTI datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.pipeline import run_case
from backend.detectors import DETECTOR_NAMES


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def nifti_stem(path: Path) -> str:
    name = path.name.lower()
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def find_cases(dataset: Path) -> list[tuple[Path, Path]]:
    """Find image/mask pairs in subject directories or a flat dataset."""
    dataset = dataset.expanduser()
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    cases = []
    for image_path in sorted(
        path for path in dataset.rglob("*")
        if path.is_file() and path.name.lower().endswith(NIFTI_SUFFIXES)
        and "mask" not in path.stem.lower()
    ):
        stem = nifti_stem(image_path)
        candidates = sorted(
            path for path in image_path.parent.iterdir()
            if path.is_file() and "mask" in path.stem.lower()
            and path.name.lower().endswith(NIFTI_SUFFIXES)
        )
        exact = [path for path in candidates if nifti_stem(path).removeprefix("mask") == stem.removeprefix("orig")]
        if len(exact) == 1:
            cases.append((image_path, exact[0]))
        elif len(candidates) == 1:
            cases.append((image_path, candidates[0]))
        else:
            raise ValueError(f"Could not identify one mask for {image_path}")
    if not cases:
        raise ValueError(f"No image/mask pairs found under {dataset}")
    return cases


def benchmark_dataset(dataset: Path, label: str, detectors: list[str]) -> list[dict[str, object]]:
    rows = []
    for image_path, mask_path in find_cases(dataset):
        row = {
            "dataset": label,
            "case_id": image_path.parent.name,
            "image": str(image_path),
            "mask": str(mask_path),
            "models": {},
        }
        for detector in detectors:
            started = time.perf_counter()
            error = ""
            daughters = ""
            try:
                result = run_case(image_path, mask_path, detector=detector)
                daughters = len(result["daughters"])
            except Exception as exc:  # Keep the timing table complete for batch runs.
                error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            row["models"][detector] = {
                "elapsed_ms": round(elapsed_ms, 3),
                "elapsed_seconds": round(elapsed_ms / 1000.0, 6),
                "daughters": daughters,
                "error": error,
            }
            print(f"{detector}/{label}/{image_path.parent.name}: {elapsed_ms / 1000.0:.3f}s"
                  + (f" ({error})" if error else ""))
        rows.append(row)
    return rows


def write_table(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({model for row in rows for model in row["models"]})
    fields = ["dataset", "case_id", "image", "mask"]
    fields.extend(f"{model}_{field}" for model in models for field in
                  ("elapsed_ms", "elapsed_seconds", "daughters", "error"))
    flat_rows = []
    for row in rows:
        flat = {field: row[field] for field in ("dataset", "case_id", "image", "mask")}
        for model in models:
            for field in ("elapsed_ms", "elapsed_seconds", "daughters", "error"):
                flat[f"{model}_{field}"] = row["models"].get(model, {}).get(field, "")
        flat_rows.append(flat)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = [result for row in rows for result in row["models"].values() if not result["error"]]
    all_values = np.asarray([float(result["elapsed_seconds"]) for result in timings])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if all_values.size:
        ax.hist(all_values, bins=min(20, max(5, int(np.sqrt(all_values.size)))), color="#2878b5", alpha=0.85)
    ax.set_title("Per-case artery detection time")
    ax.set_xlabel("Elapsed time (seconds)")
    ax.set_ylabel("Number of cases")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "detection_time_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    if timings:
        ax.boxplot(all_values, tick_labels=["All cases"], patch_artist=True,
                   boxprops={"facecolor": "#72b7b2", "alpha": 0.85},
                   medianprops={"color": "#d95f02", "linewidth": 2})
    ax.set_title("Per-case artery detection time")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "detection_time_boxplot.png", dpi=150)
    plt.close(fig)
    old_barplot = output_dir / "detection_time_barplot.png"
    if old_barplot.exists():
        old_barplot.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("LABEL", "PATH"),
                        help="Dataset label and path; repeat to benchmark additional datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("timing_results"))
    parser.add_argument("--detector", choices=DETECTOR_NAMES,
                        help="Run one detector; omit to benchmark every registered detector")
    args = parser.parse_args(argv)

    datasets = args.dataset or [["dataset", "dataset"], ["synthetic_dataset", "synthetic_dataset"]]
    detectors = [args.detector] if args.detector else list(DETECTOR_NAMES)
    rows = []
    for label, path in datasets:
        rows.extend(benchmark_dataset(Path(path), label, detectors))
    write_table(rows, args.output_dir / "detection_times.csv")
    (args.output_dir / "detection_times.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_plots(rows, args.output_dir)

    successful = [
        float(result["elapsed_seconds"])
        for row in rows
        for result in row["models"].values()
        if not result["error"]
    ]
    print(f"Wrote {len(rows)} case datapoints to {args.output_dir}")
    if successful:
        print(f"Successful cases: {len(successful)}; mean={np.mean(successful):.3f}s; median={np.median(successful):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
