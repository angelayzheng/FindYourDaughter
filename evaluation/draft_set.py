"""Read the eval_set package and evaluate registered detectors without reference input."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import time

import numpy as np
import SimpleITK as sitk

from backend.detectors import DETECTOR_NAMES, detect
from backend.configuration import configuration
from backend.inputs import CaseData, _same_geometry, load_case
from evaluation.landmarks import score_landmarks, summarize, usable_radius, validate_branches


SCOPE = ("Agreement with draft, potentially incomplete annotations; not official challenge scoring "
         "or expert-validated accuracy. Unmatched predictions require review, not automatic false-positive labels.")


@dataclass
class ReferenceCase:
    case: CaseData
    annotation: dict
    labels: sitk.Image
    files: tuple[Path, ...]


def read_reference(directory: Path) -> ReferenceCase:
    """Validate the package's parent-only input and separate daughter references."""
    match = re.fullmatch(r"case_(\d+)", directory.name)
    if match is None:
        raise ValueError(f"Expected a case_NUMBER directory: {directory}")
    number = int(match[1])
    annotation_path = directory / "annotations.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotation.get("case_id") != directory.name:
        raise ValueError("Annotation case_id does not match its directory")
    if annotation.get("coordinate_system") != "SimpleITK physical LPS millimetres":
        raise ValueError("Annotations must explicitly use SimpleITK physical LPS millimetres")
    if annotation.get("parent", {}).get("instance_id") != "aorta":
        raise ValueError("Annotation parent must be aorta")
    references = annotation.get("daughters")
    validate_branches(references, reference=True)
    label_values = [r.get("label_value") for r in references]
    if (any(type(label) is not int or label <= 0 for label in label_values)
            or len(set(label_values)) != len(label_values)):
        raise ValueError("Reference label_value fields must be unique positive integers")
    image_path, mask_path = directory / f"orig{number}.nii.gz", directory / f"aorta{number}.nii.gz"
    label_path = directory / f"daughters{number}_draft.nii.gz"
    case = load_case(image_path, mask_path)
    labels = sitk.ReadImage(str(label_path))
    if (labels.GetDimension() != 3 or labels.GetNumberOfComponentsPerPixel() != 1
            or not _same_geometry(case.image, labels)):
        raise ValueError("Daughter reference mask must match the CT's scalar 3-D physical geometry")
    if (list(case.image.GetSize()) != annotation.get("shape_xyz")
            or not np.allclose(case.image.GetSpacing(), annotation.get("spacing_xyz_mm"), atol=1e-6, rtol=0)):
        raise ValueError("Annotation shape/spacing metadata does not match the CT")
    parent = sitk.GetArrayViewFromImage(case.aorta_mask)
    if not np.all((parent == 0) | (parent == 1)):
        raise ValueError("Detector input must be the binary parent-only aorta mask")
    data = sitk.GetArrayViewFromImage(labels)
    actual = np.unique(data)
    if not np.isfinite(actual).all() or set(actual.tolist()) - {0} != set(label_values):
        raise ValueError("Daughter mask labels do not match annotation label_value fields")
    if np.any((data != 0) & (parent != 0)):
        raise ValueError("Daughter reference mask overlaps the supplied parent")
    for branch in references:
        for key in ("ostium_xyz_mm", "seed_xyz_mm"):
            index = case.image.TransformPhysicalPointToContinuousIndex(branch[key])
            if not all(-.5 <= index[k] < case.image.GetSize()[k] - .5 for k in range(3)):
                raise ValueError(f"Reference {branch['instance_id']} {key} lies outside the CT")
        # Check the redundant voxel/physical guides when supplied. This catches
        # accidental RAS or XYZ/ZYX use without deriving truth from a detector.
        if "centerline_voxel_xyz" in branch and "centerline_xyz_mm" in branch:
            physical = np.asarray([case.image.TransformContinuousIndexToPhysicalPoint(point)
                                   for point in branch["centerline_voxel_xyz"]])
            supplied = np.asarray(branch["centerline_xyz_mm"], dtype=float)
            if physical.shape != supplied.shape or not np.allclose(physical, supplied, atol=1e-3, rtol=0):
                raise ValueError("Reference voxel guides disagree with physical LPS guides")
    return ReferenceCase(case, annotation, labels, (annotation_path, image_path, mask_path, label_path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _markdown(report: dict) -> str:
    lines = ["# Draft reference evaluation", "", SCOPE, "",
             f"Status: **{report['status']}**. Ostium matching tolerance: **{report['matching_tolerance_mm']:g} mm**.",
             "Effective parameters are recorded in report.json; only CT, the parent mask, and those parameters enter detection.", "",
             "| Detector | Completed cases | Matches / references | Unmatched predictions | Reference precision | Reference recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]

    def percent(value):
        return f"{100 * value:.1f}%" if value is not None else "n/a"

    for name, summary in report["summary"].items():
        lines.append(f"| {name} | {summary['successful_cases']}/{report['case_count']} | "
                     f"{summary['matched']}/{summary['reference_count']} | {summary['unmatched_predictions']} | "
                     f"{percent(summary['reference_precision'])} | {percent(summary['reference_recall'])} |")
    lines += ["", "| Case | Detector | Matches / references | Unmatched predictions | Inference seconds |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in report["cases"]:
        for name in report["detectors"]:
            run = row.get("detectors", {}).get(name, {})
            if row["status"] == "error" or run.get("status") != "ok":
                lines.append(f"| {row['case_id']} | {name} | ERROR | n/a | n/a |")
            else:
                summary = run["scores"]["summary"]
                lines.append(f"| {row['case_id']} | {name} | {summary['matched']}/{summary['reference_count']} | "
                             f"{summary['unmatched_predictions']} | {run['inference_seconds']:.3f} |")
    lines += ["", "`report.json` contains matched IDs, landmark errors, unmatched IDs and nearest-reference distances,",
              "native label checks, review flags, input hashes, failures, and timing details.",
              "Null measurements and empty denominators remain null. Radius error uses only usable matched radius references.",
              "Failed cases are excluded from metric denominators and counted explicitly; an incomplete run is not a full-set score.",
              "Predictions and detector diagnostics are exported separately under each detector's directory.", ""]
    return "\n".join(lines)


def evaluate_dataset(dataset: Path, output: Path, *, detectors: tuple[str, ...] = DETECTOR_NAMES,
                     tolerance_mm: float = 3., cases: list[int] | None = None,
                     parameters: dict | None = None) -> dict:
    """Run cases serially, record failures, and save a reviewable paired report."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset):
        raise ValueError("Evaluation output must be outside the reference dataset")
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Matching tolerance must be finite and positive")
    if not detectors or len(set(detectors)) != len(detectors) or any(d not in DETECTOR_NAMES for d in detectors):
        raise ValueError(f"Select unique registered detectors from {DETECTOR_NAMES}")
    if parameters is not None and (not isinstance(parameters, dict) or set(parameters) - set(detectors)):
        raise ValueError("Parameters must be keyed by selected detector names")
    configurations = {name: configuration(name, (parameters or {}).get(name)) for name in detectors}
    directories = sorted((p for p in dataset.glob("case_*") if p.is_dir()), key=lambda p: p.name)
    if not directories:
        raise ValueError(f"No case_NUMBER directories found in {dataset}")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if cases is None and "case_counts" in manifest:
        expected = {f"case_{int(number)}" for number in manifest["case_counts"]}
        actual = {p.name for p in directories}
        if expected != actual:
            raise ValueError(f"Case inventory differs from manifest: missing {sorted(expected - actual)}, extra {sorted(actual - expected)}")
    if cases is not None:
        requested = {f"case_{number}" for number in cases}
        missing = requested - {p.name for p in directories}
        if not requested or missing:
            raise ValueError(f"Requested cases not found: {sorted(missing)}")
        directories = [p for p in directories if p.name in requested]
    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    report = {
        "scope": SCOPE, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "annotation_status": manifest.get("annotation_status", "unspecified"),
        "manifest_sha256": _sha256(manifest_path) if manifest_path.exists() else None,
        "matching_tolerance_mm": tolerance_mm, "matching": "one-to-one maximum cardinality then minimum LPS ostium distance",
        "detectors": list(detectors), "detector_settings": "explicit parameters" if parameters else "unchanged registry defaults",
        "configurations": configurations,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "SimpleITK", "scikit-image", "nibabel")},
        "backend_sha256": {p.name: _sha256(p) for p in sorted(backend_dir.glob("*.py"))},
        "evaluation_sha256": {p.name: _sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "case_count": len(directories), "cases": [], "summary": {},
    }
    for directory in directories:
        row = {"case_id": directory.name, "status": "ok", "detectors": {}}
        report["cases"].append(row)
        try:
            start = time.perf_counter()
            reference = read_reference(directory)
            row["load_and_validation_seconds"] = time.perf_counter() - start
            row["annotation_status"] = reference.annotation.get("annotation_status", "unspecified")
            branches = reference.annotation["daughters"]
            row["reference_count"] = len(branches)
            row["usable_radius_reference_count"] = sum(usable_radius(b) for b in branches)
            row["input_sha256"] = {p.relative_to(dataset).as_posix(): _sha256(p) for p in reference.files}
            row["review_notes"] = (directory / "review_notes.md").relative_to(dataset).as_posix() if (directory / "review_notes.md").exists() else None
            row["excluded_candidate_present"] = (directory / "excluded_candidate.json").exists()
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
            report["status"] = "incomplete"
            continue
        for name in detectors:
            try:
                start = time.perf_counter()
                # Reference labels, guides, per-case thresholds and landmarks
                # are intentionally not detector inputs or parameter settings.
                settings = {"parameters": configurations[name]["parameters"]} if parameters is not None else {}
                result = detect(reference.case.image, reference.case.aorta_mask, detector=name, **settings)
                elapsed = time.perf_counter() - start
                prediction = {"case_id": directory.name, "parent": {"instance_id": "aorta"},
                              "daughters": result.daughters()}
                scores = score_landmarks(prediction["daughters"], branches, tolerance_mm=tolerance_mm,
                                         labels=reference.labels)
                _write_json(output / name / f"{directory.name}_prediction.json", prediction)
                _write_json(output / name / f"{directory.name}_diagnostics.json", result.diagnostics)
                row["detectors"][name] = {"status": "ok", "inference_seconds": elapsed, "scores": scores}
            except Exception as error:
                row["detectors"][name] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
                report["status"] = "incomplete"
        del reference
    for name in detectors:
        successful = [row["detectors"][name] for row in report["cases"]
                      if row["status"] == "ok" and row["detectors"].get(name, {}).get("status") == "ok"]
        matches = [match for run in successful for match in run["scores"]["matches"]]
        summary = summarize(matches, sum(r["scores"]["summary"]["prediction_count"] for r in successful),
                            sum(r["scores"]["summary"]["reference_count"] for r in successful))
        summary.update(successful_cases=len(successful), failed_cases=len(directories) - len(successful),
                       mean_inference_seconds=float(np.mean([r["inference_seconds"] for r in successful])) if successful else None)
        report["summary"][name] = summary
    _write_json(output / "report.json", report)
    (output / "summary.md").write_text(_markdown(report), encoding="utf-8")
    return report
