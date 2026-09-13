"""Scoring contract tests with independent small fixtures, no eval_set dependency."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from backend.detection import DetectionResult
from evaluation.draft_set import evaluate_dataset, read_reference
from evaluation.landmarks import score_landmarks, summarize
from scripts.evaluate_detectors import main
from test_detection import phantom


def branch(identifier, x=0., *, radius=2., label=1):
    return {"instance_id": identifier, "parent_instance_id": "aorta", "label_value": label,
            "ostium_xyz_mm": [x, 0., 0.], "seed_xyz_mm": [x + 5, 0., 0.],
            "direction_xyz": [1., 0., 0.], "radius_mm": radius,
            "review_status": "expert_review_pending"}


class LandmarkScoringTest(unittest.TestCase):
    def test_matching_maximizes_cardinality_and_ignores_instance_id_equality(self):
        # Greedily matching p1 to r1 loses a valid match for p2. The names are
        # intentionally equal across sets, but the physical correspondence differs.
        predictions = [branch("branch_001", 1.5), branch("branch_002", -2.5)]
        references = [branch("branch_001", 0), branch("branch_002", 4)]
        result = score_landmarks(predictions, references)
        self.assertEqual(result["summary"]["matched"], 2)
        self.assertEqual([(m["prediction_id"], m["reference_id"]) for m in result["matches"]],
                         [("branch_001", "branch_002"), ("branch_002", "branch_001")])

    def test_duplicate_proposals_do_not_get_duplicate_credit(self):
        result = score_landmarks([branch("p1"), branch("p2", .1)], [branch("r")])
        self.assertEqual(result["summary"]["matched"], 1)
        self.assertEqual(result["summary"]["reference_precision"], .5)
        self.assertEqual(result["summary"]["reference_recall"], 1)
        self.assertEqual(result["unmatched_predictions"][0]["prediction_id"], "p2")
        self.assertAlmostEqual(result["unmatched_predictions"][0]["nearest_ostium_distance_mm"], .1)

    def test_matching_tolerance_is_inclusive_and_configurable(self):
        self.assertEqual(score_landmarks([branch("p", 3)], [branch("r")])["summary"]["matched"], 1)
        result = score_landmarks([branch("p", 3.001)], [branch("r")])
        self.assertEqual(result["summary"]["matched"], 0)
        self.assertEqual(result["unmatched_reference_ids"], ["r"])
        self.assertEqual(score_landmarks([branch("p", 3.001)], [branch("r")], tolerance_mm=4)["summary"]["matched"], 1)

    def test_empty_sets_have_explicit_counts_and_undefined_denominators(self):
        for predictions, references in (([], []), ([], [branch("r")]), ([branch("p")], [])):
            with self.subTest(predictions=len(predictions), references=len(references)):
                result = score_landmarks(predictions, references)
                self.assertEqual(result["summary"]["matched"], 0)
                self.assertEqual(result["summary"]["unmatched_predictions"], len(predictions))
                self.assertEqual(result["summary"]["unmatched_references"], len(references))
                self.assertIsNone(result["summary"]["errors"]["seed_error_mm"]["mean"])
                if not predictions:
                    self.assertIsNone(result["summary"]["reference_precision"])
                if not references:
                    self.assertIsNone(result["summary"]["reference_recall"])
                json.dumps(result, allow_nan=False)

    def test_missing_or_limited_radii_are_excluded_without_using_origin_diameter(self):
        for radius, limited in ((None, False), (2., True)):
            reference = branch("r", radius=radius)
            reference.update(origin_diameter_estimate_mm=8., seed_cross_section_touches_search_limit=limited)
            result = score_landmarks([branch("p")], [reference])
            self.assertEqual(result["summary"]["errors"]["radius_error_mm"]["count"], 0)
            self.assertIsNone(result["matches"][0]["radius_error_mm"])
        reference = branch("r", radius=1.2)
        reference["radius_measurement_status"] = "approximate_threshold_estimate"
        predicted = branch("p", radius=2.)
        predicted["direction_xyz"] = [-1., 0., 0.]
        result = score_landmarks([predicted], [reference])
        self.assertAlmostEqual(result["matches"][0]["radius_error_mm"], .8)
        self.assertEqual(result["matches"][0]["direction_error_deg"], 180.)

    def test_native_label_lookup_uses_lps_geometry_and_handles_outside_seeds(self):
        labels = sitk.Image([4, 5, 6], sitk.sitkUInt8)
        labels.SetSpacing((2., 3., 4.))
        labels.SetOrigin((17., -31., 80.))
        labels.SetDirection((0., -1., 0., -1., 0., 0., 0., 0., 1.))
        labels[1, 2, 3] = 7
        seed = labels.TransformIndexToPhysicalPoint((1, 2, 3))
        reference = branch("r", label=7)
        reference["seed_xyz_mm"] = list(seed)
        predicted = deepcopy(reference)
        predicted["instance_id"] = "p"
        match = score_landmarks([predicted], [reference], labels=labels)["matches"][0]
        self.assertTrue(match["seed_in_matched_label"])
        self.assertEqual(match["seed_distance_to_label_voxel_mm"], 0)
        predicted["seed_xyz_mm"] = list(labels.TransformIndexToPhysicalPoint((2, 2, 3)))
        match = score_landmarks([predicted], [reference], labels=labels)["matches"][0]
        self.assertFalse(match["seed_in_matched_label"])
        self.assertAlmostEqual(match["seed_distance_to_label_voxel_mm"], 2.)
        predicted["seed_xyz_mm"] = [1000., 0., 0.]
        match = score_landmarks([predicted], [reference], labels=labels)["matches"][0]
        self.assertFalse(match["seed_in_matched_label"])
        self.assertIsNone(match["seed_nearest_label_value"])

    def test_invalid_predictions_references_and_tolerances_fail(self):
        for key, value in (("ostium_xyz_mm", [np.nan, 0, 0]), ("seed_xyz_mm", [1, 2]),
                           ("direction_xyz", [0, 0, 0]), ("radius_mm", None), ("radius_mm", -1)):
            predicted = branch("p")
            predicted[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                score_landmarks([predicted], [branch("r")])
        with self.assertRaisesRegex(ValueError, "unique"):
            score_landmarks([branch("p"), branch("p")], [])
        for tolerance in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                score_landmarks([], [], tolerance_mm=tolerance)

    def test_aggregate_is_micro_averaged_and_errors_only_use_matches(self):
        first = score_landmarks([branch("p")], [branch("r")])
        second = score_landmarks([], [branch(f"r{i}", i * 10) for i in range(9)])
        summary = summarize(first["matches"] + second["matches"], 1, 10)
        self.assertAlmostEqual(summary["reference_recall"], .1)
        self.assertEqual(summary["errors"]["ostium_error_mm"]["count"], 1)


class DraftSetTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "reference"
        self.directory = self.dataset / "case_1"
        self.directory.mkdir(parents=True)
        image, parent = phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        data = sitk.GetArrayFromImage(image)
        x = np.indices(data.shape)[2]
        label = ((data > 100) & (x > 37)).astype(np.uint8) * 7
        labels = sitk.GetImageFromArray(label)
        labels.CopyInformation(image)
        for filename, volume in (("orig1.nii.gz", image), ("aorta1.nii.gz", parent), ("daughters1_draft.nii.gz", labels)):
            sitk.WriteImage(volume, str(self.directory / filename))
        reference = branch("draft_branch", radius=None, label=7)
        reference.update(ostium_xyz_mm=[54.5, 1., 116.], seed_xyz_mm=[59.5, 1., 116.])
        self.annotation = {"case_id": "case_1", "coordinate_system": "SimpleITK physical LPS millimetres",
                           "parent": {"instance_id": "aorta"}, "shape_xyz": list(image.GetSize()),
                           "spacing_xyz_mm": list(image.GetSpacing()), "annotation_status": "expert_review_pending",
                           "daughters": [reference]}
        self.write_annotation()

    def write_annotation(self):
        (self.directory / "annotations.json").write_text(json.dumps(self.annotation), encoding="utf-8")

    def test_cli_runs_both_detectors_and_keeps_reference_files_unchanged(self):
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        output = self.root / "results"
        self.assertEqual(main(["--dataset", str(self.dataset), "--output-dir", str(output)]), 0)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "complete")
        for name in ("baseline", "contact"):
            self.assertEqual(report["summary"][name]["matched"], 1, report)
            self.assertEqual(report["summary"][name]["errors"]["radius_error_mm"]["count"], 0)
            prediction = json.loads((output / name / "case_1_prediction.json").read_text())
            self.assertEqual(prediction["case_id"], "case_1")
            self.assertEqual(set(prediction), {"case_id", "parent", "daughters"})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})
        self.assertTrue((output / "summary.md").is_file())

    def test_only_ct_and_parent_are_given_to_detector(self):
        def detector(image, parent, *, detector):
            self.assertEqual(detector, "baseline")
            self.assertEqual(set(np.unique(sitk.GetArrayFromImage(parent))), {0, 1})
            self.assertNotIn(7, np.unique(sitk.GetArrayFromImage(parent)))
            return DetectionResult()
        with patch("evaluation.draft_set.detect", side_effect=detector) as mocked:
            report = evaluate_dataset(self.dataset, self.root / "result", detectors=("baseline",))
        mocked.assert_called_once()
        self.assertEqual(report["summary"]["baseline"]["unmatched_references"], 1)

    def test_failed_detector_is_recorded_without_erasing_other_results(self):
        def detector(image, parent, *, detector):
            if detector == "contact":
                raise RuntimeError("test failure")
            return DetectionResult()
        with patch("evaluation.draft_set.detect", side_effect=detector):
            code = main(["--dataset", str(self.dataset), "--output-dir", str(self.root / "results")])
        report = json.loads((self.root / "results/report.json").read_text())
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["summary"]["baseline"]["successful_cases"], 1)
        self.assertEqual(report["summary"]["contact"]["failed_cases"], 1)
        self.assertEqual(report["summary"]["contact"]["reference_count"], 0)
        self.assertIn("test failure", report["cases"][0]["detectors"]["contact"]["error"])

    def test_missing_reference_files_are_not_silently_skipped(self):
        (self.directory / "annotations.json").unlink()
        report = evaluate_dataset(self.dataset, self.root / "results")
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["cases"][0]["status"], "error")
        self.assertEqual(report["summary"]["baseline"]["failed_cases"], 1)

    def test_invalid_reference_coordinates_labels_and_parent_are_rejected(self):
        original = deepcopy(self.annotation)
        for key, value in (("coordinate_system", "RAS"), ("case_id", "case_99"), ("shape_xyz", [1, 2, 3])):
            with self.subTest(key=key):
                self.annotation = deepcopy(original)
                self.annotation[key] = value
                self.write_annotation()
                with self.assertRaises(ValueError):
                    read_reference(self.directory)
        self.annotation = original
        self.annotation["daughters"][0]["label_value"] = 8
        self.write_annotation()
        with self.assertRaisesRegex(ValueError, "labels"):
            read_reference(self.directory)
        self.annotation["daughters"][0]["label_value"] = 7
        self.write_annotation()
        path = self.directory / "aorta1.nii.gz"
        mask = sitk.ReadImage(str(path))
        sitk.WriteImage(mask * 2, str(path))
        with self.assertRaisesRegex(ValueError, "binary parent-only"):
            read_reference(self.directory)

    def test_label_geometry_mismatch_is_not_resampled_away(self):
        path = self.directory / "daughters1_draft.nii.gz"
        labels = sitk.ReadImage(str(path))
        labels.SetOrigin((18., -31., 80.))
        sitk.WriteImage(labels, str(path))
        with self.assertRaisesRegex(ValueError, "geometry"):
            read_reference(self.directory)

    def test_output_cannot_overwrite_reference_and_unknown_cases_fail(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluate_dataset(self.dataset, self.dataset / "results")
        with self.assertRaisesRegex(ValueError, "not found"):
            evaluate_dataset(self.dataset, self.root / "results", cases=[99])
        with self.assertRaisesRegex(ValueError, "No case"):
            evaluate_dataset(self.root / "empty", self.root / "results")

    def test_manifest_inventory_detects_missing_case_directories(self):
        (self.dataset / "manifest.json").write_text(json.dumps({"case_counts": {"1": 1, "2": 1}}))
        with self.assertRaisesRegex(ValueError, "inventory"):
            evaluate_dataset(self.dataset, self.root / "results")
        # An explicitly requested available subset is still useful.
        with patch("evaluation.draft_set.detect", return_value=DetectionResult()):
            report = evaluate_dataset(self.dataset, self.root / "results", cases=[1], detectors=("contact",))
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["detectors"], ["contact"])


if __name__ == "__main__":
    unittest.main()
