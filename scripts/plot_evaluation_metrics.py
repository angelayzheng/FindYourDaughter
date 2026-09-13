"""Plot compact detector comparisons from saved evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DETECTORS = ("baseline", "contact", "fusion", "refined")
DATASETS = ("Real draft", "Fresh synthetic")
COLORS = {"Real draft": "#2f6690", "Fresh synthetic": "#f28e2b"}


def read_reports(real_report: Path, synthetic_directory: Path) -> dict[str, dict[str, dict]]:
    real = json.loads(real_report.read_text(encoding="utf-8"))["summary"]
    synthetic = {}
    for detector in DETECTORS:
        path = synthetic_directory / f"{detector}.json"
        synthetic[detector] = json.loads(path.read_text(encoding="utf-8"))
    return {"Real draft": real, "Fresh synthetic": synthetic}


def metric(report: dict, name: str) -> float:
    if name in ("precision", "recall", "f1"):
        return float(report[f"reference_{name}"] if "reference_" + name in report else {
            "precision": report["true_positive"] / (report["true_positive"] + report["false_positive"]),
            "recall": report["true_positive"] / (report["true_positive"] + report["false_negative"]),
            "f1": 2 * report["true_positive"] / (2 * report["true_positive"] + report["false_positive"] + report["false_negative"]),
        }[name])
    if name == "matched":
        return float(report["matched"] if "matched" in report else report["true_positive"])
    if name == "missed":
        return float(report["unmatched_references"] if "unmatched_references" in report else report["false_negative"])
    if name == "extras":
        return float(report["unmatched_predictions"] if "unmatched_predictions" in report else report["false_positive"])
    error_name = {"ostium": "ostium_error_mm", "seed": "seed_error_mm", "radius": "radius_error_mm"}[name]
    if "errors" in report:
        return float(report["errors"][error_name]["mean"])
    return float(report["mean_errors"][f"{name}_mm"])


def plot(reports: dict[str, dict[str, dict]], output: Path) -> None:
    labels = np.arange(len(DETECTORS))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.18, top=0.78, wspace=0.28)
    figure.suptitle("Detector performance: real draft vs fresh synthetic", fontsize=14, fontweight="bold")

    score_axis = axes[0]
    score_names = ("precision", "recall", "f1")
    score_labels = ("Precision", "Recall", "F1")
    score_colors = ("#264653", "#e9c46a", "#e76f51")
    for name, label, color in zip(score_names, score_labels, score_colors):
        for dataset, linestyle in zip(DATASETS, ("-", "--")):
            values = [metric(reports[dataset][detector], name) for detector in DETECTORS]
            score_axis.plot(labels, values, marker="o", linewidth=2, linestyle=linestyle, color=color,
                            markerfacecolor=COLORS[dataset], markeredgecolor=color,
                            label=f"{label}, {dataset}")
    score_axis.set_title("Agreement scores")
    score_axis.set_ylabel("Score")
    score_axis.set_ylim(0, 1.05)
    score_axis.set_xticks(labels, DETECTORS)
    score_axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    score_axis.grid(axis="y", alpha=0.25)
    score_axis.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)

    count_axis = axes[1]
    width = 0.32
    for dataset_index, dataset in enumerate(DATASETS):
        for metric_index, name in enumerate(("matched", "missed", "extras")):
            values = [metric(reports[dataset][detector], name) for detector in DETECTORS]
            position = labels + (dataset_index - 0.5) * width
            bottom = [0.0] * len(DETECTORS)
            for previous in ("matched", "missed", "extras")[:metric_index]:
                bottom = [base + value for base, value in zip(bottom, [metric(reports[dataset][detector], previous) for detector in DETECTORS])]
            count_axis.bar(position, values, width=width, bottom=bottom, color=("#457b9d", "#e9c46a", "#e76f51")[metric_index],
                           edgecolor=COLORS[dataset], hatch="///" if dataset_index else None,
                           linewidth=0.8, label=f"{name.title()} ({dataset})")
    count_axis.set_title("Detection counts")
    count_axis.set_ylabel("Daughters")
    count_axis.set_xticks(labels, DETECTORS)
    count_axis.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")

    error_axis = axes[2]
    error_names = ("ostium", "seed", "radius")
    error_labels = ("Ostium", "Seed", "Radius")
    error_colors = ("#457b9d", "#6a994e", "#bc6c25")
    for name, label, color in zip(error_names, error_labels, error_colors):
        for dataset, linestyle in zip(DATASETS, ("-", "--")):
            values = [metric(reports[dataset][detector], name) for detector in DETECTORS]
            error_axis.plot(labels, values, marker="o", linewidth=2, linestyle=linestyle, color=color,
                            markerfacecolor=COLORS[dataset], markeredgecolor=color,
                            label=f"{label}, {dataset}")
    error_axis.set_title("Mean matched error")
    error_axis.set_ylabel("Millimetres")
    error_axis.set_xticks(labels, DETECTORS)
    error_axis.grid(axis="y", alpha=0.25)
    error_axis.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    count_axis.text(0.02, 1.02, "Hatched = fresh synthetic", transform=count_axis.transAxes, fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-report", type=Path, default=Path("nifti_previews/improvement_refined/report.json"))
    parser.add_argument("--synthetic-directory", type=Path, default=Path("tmp/synthetic-full-20260913"))
    parser.add_argument("--output", type=Path, default=Path("nifti_previews/evaluation_model_comparison.png"))
    args = parser.parse_args()
    plot(read_reports(args.real_report, args.synthetic_directory), args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())