"""Plot compact detector comparisons from saved evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DETECTORS = ("baseline", "contact", "fusion", "refined")
DATASETS = ("Real draft", "Fresh synthetic")


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
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="row")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.88, hspace=0.62, wspace=0.18)
    figure.suptitle("Detector performance by evaluation set", fontsize=16, fontweight="bold")

    score_names = ("precision", "recall", "f1")
    score_labels = ("Precision", "Recall", "F1")
    score_colors = ("#264653", "#e9c46a", "#e76f51")
    count_names = ("matched", "missed", "extras")
    count_labels = ("Matched", "Missed", "Extras")
    count_colors = ("#457b9d", "#e9c46a", "#e76f51")
    error_names = ("ostium", "seed", "radius")
    error_labels = ("Ostium", "Seed", "Radius")
    error_colors = ("#457b9d", "#6a994e", "#bc6c25")

    for dataset_index, dataset in enumerate(DATASETS):
        score_axis, count_axis, error_axis = axes[:, dataset_index]
        dataset_report = reports[dataset]
        reference_count = int(dataset_report[DETECTORS[0]].get("reference_count", 48))
        score_width = 0.24
        for metric_index, (name, label, color) in enumerate(zip(score_names, score_labels, score_colors)):
            values = [metric(dataset_report[detector], name) for detector in DETECTORS]
            score_axis.bar(labels + (metric_index - 1) * score_width, values, width=score_width,
                           color=color, label=label)
        score_axis.set_title(f"{dataset}: agreement scores\n({reference_count} references)", pad=28)
        score_axis.set_ylim(0, 1.05)
        score_axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        score_axis.grid(axis="y", alpha=0.25)
        score_axis.set_axisbelow(True)
        score_axis.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

        count_width = 0.24
        for metric_index, (name, label, color) in enumerate(zip(count_names, count_labels, count_colors)):
            values = [metric(dataset_report[detector], name) for detector in DETECTORS]
            count_axis.bar(labels + (metric_index - 1) * count_width, values, width=count_width,
                           color=color, label=label)
        count_axis.set_title(f"{dataset}: outcome counts\n({reference_count} references)", pad=28)
        count_axis.grid(axis="y", alpha=0.25)
        count_axis.set_axisbelow(True)
        count_axis.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

        error_width = 0.24
        for metric_index, (name, label, color) in enumerate(zip(error_names, error_labels, error_colors)):
            values = [metric(dataset_report[detector], name) for detector in DETECTORS]
            error_axis.bar(labels + (metric_index - 1) * error_width, values, width=error_width,
                           color=color, label=label)
        radius_note = "radius coverage limited" if dataset == "Real draft" else "all matched radius values"
        error_axis.set_title(f"{dataset}: mean matched error\n({radius_note})", pad=28)
        error_axis.grid(axis="y", alpha=0.25)
        error_axis.set_axisbelow(True)
        error_axis.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)

    for axis in axes.flat:
        axis.set_xticks(labels, DETECTORS)
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Daughters")
    axes[2, 0].set_ylabel("Millimetres")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[2, 1].set_ylabel("")
    figure.text(0.5, 0.015, "Errors are computed only for matched branches; radius coverage is limited in the real draft set.",
                ha="center", fontsize=8, color="#555555")
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