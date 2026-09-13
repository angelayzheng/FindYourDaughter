"""Regression fixtures for the opt-in detector, without real reference input."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import test_detection as anatomy
from test_fusion_detection import candidate
import test_fusion_detection as fusion_tests
from backend.detectors import detect
from backend.fusion_detection import _EvidenceVolume, detect_fusion
from backend.refined_detection import _extended_caps, _repair_parent_chord, _selection_key, detect_refined


class RefinedGeometryTest(anatomy.ArteryDetectionTest):
    def setUp(self):
        replacement = patch.object(anatomy, "detect_daughters", detect_refined)
        replacement.start()
        self.addCleanup(replacement.stop)


class RefinedStressTest(unittest.TestCase):
    def setUp(self):
        replacement = patch.object(fusion_tests, "detect_fusion", detect_refined)
        replacement.start()
        self.addCleanup(replacement.stop)

    test_weak_noisy_branch = fusion_tests.FusionCandidateTest.test_low_contrast_noisy_branch_is_retained_and_inputs_are_unchanged
    test_early_common_trunk = fusion_tests.FusionCandidateTest.test_early_common_trunk_is_retained_for_review_without_child_substitution
    test_bright_pool = fusion_tests.FusionCandidateTest.test_bright_pool_on_a_thin_bridge_is_not_a_daughter
    test_coarse_spacing = fusion_tests.FusionCandidateTest.test_coarse_through_plane_spacing_preserves_physical_coordinates


class RefinedDetectionTest(unittest.TestCase):
    def test_retracted_endpoint_reaches_observed_cut_but_preserves_lateral_opening(self):
        image, mask = anatomy.phantom([((32, 32, 14), (64, 32, 14), 1.7)])
        volume = _EvidenceVolume(image, mask, {"threshold": 120., "blood_intensity": 300.})
        # An endpoint 13 mm inside the cut, with a smaller 5.5 mm EDT radius:
        # the old radius-based ball cannot reach the crop plane.
        center = np.array(image.TransformIndexToPhysicalPoint((32, 32, 25)))
        volume.caps = [(center, np.array([0., 0., -1.]), 8.5)]
        cap = _extended_caps(volume)[0]
        cut = np.array(image.TransformContinuousIndexToPhysicalPoint((32., 32., 11.5)))
        self.assertLessEqual(np.linalg.norm(cut - cap[0]), cap[2])
        # Integration also keeps a daughter immediately beside a cropped end.
        result = detect_refined(image, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        self.assertGreater(result.branches[0].direction_xyz[0], .9)

    @staticmethod
    def concave_wall():
        z, y, x = np.indices((32, 32, 32))
        parent = x <= 10
        parent[(x == 8) & (abs(y - 16) <= 1) & (abs(z - 16) <= 1)] = False
        tube = ((y - 16)**2 + (z - 16)**2 <= 2.5**2)
        image = sitk.GetImageFromArray(np.where(parent | tube, 300., 30.).astype(np.float32))
        mask = sitk.GetImageFromArray(parent.astype(np.uint8))
        image.SetOrigin((17., -31., 80.))
        image.SetSpacing((.8, 1., 1.2))
        image.SetDirection((0., -1., 0., -1., 0., 0., 0., 0., 1.))
        mask.CopyInformation(image)
        path = np.asarray([image.TransformContinuousIndexToPhysicalPoint((v, 16., 16.))
                           for v in (8.5, 12., 21.)])
        return image, mask, path

    def test_parent_chord_repair_uses_physical_exit_and_retains_external_path(self):
        image, mask, path = self.concave_wall()
        volume = _EvidenceVolume(image, mask, {"threshold": 120., "blood_intensity": 300.})
        result, reason = _repair_parent_chord(path, volume)
        self.assertEqual(reason, "repaired_parent_chord")
        expected = image.TransformContinuousIndexToPhysicalPoint((10.5, 16., 16.))
        np.testing.assert_allclose(result[0], expected, atol=1e-4)
        np.testing.assert_array_equal(result[1:], path[1:])
        self.assertGreaterEqual(np.linalg.norm(np.diff(result, axis=0), axis=1).sum(), 5.)

    def test_repair_never_accepts_a_later_return_to_parent_or_a_short_trunk(self):
        image, mask, path = self.concave_wall()
        volume = _EvidenceVolume(image, mask, {"threshold": 120., "blood_intensity": 300.})
        returned = np.vstack([path, path[0]])
        self.assertEqual(_repair_parent_chord(returned, volume)[1], "unsupported_repaired_path")
        self.assertEqual(_repair_parent_chord(path[:2], volume)[1], "short_repaired_path")

    def test_naturally_closed_sections_resolve_duplicate_parent_contamination(self):
        clean, contaminated = candidate("baseline"), candidate("contact")
        for c in (clean, contaminated):
            c.features.update(radius_quality="stable_sections", review_score=.9,
                              sections=[{"closed": True, "method": "orthogonal_section"} for _ in range(3)])
        clean.features["review_score"] = .6
        contaminated.features["sections"][0]["method"] = "parent_excluded_section"
        self.assertIs(max([clean, contaminated], key=_selection_key), clean)

    def test_original_models_are_unchanged_after_running_refinement(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        before = {name: detect(image, mask, detector=name).daughters() for name in ("baseline", "contact", "fusion")}
        result = detect(image, mask, detector="refined")
        self.assertEqual(result.diagnostics["detector"], "refined")
        json.dumps(result.diagnostics, allow_nan=False)
        self.assertEqual(len(result.branches), 1)
        for name, predictions in before.items():
            self.assertEqual(predictions, detect(image, mask, detector=name).daughters())

    def test_open_section_proposals_remain_auditable_but_are_not_exported(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        source = detect_fusion(image, mask)
        from backend.detection import DetectionResult
        # A contrast-filled broad sheet across each section has no closed
        # daughter lumen, even when its connecting source path was accepted.
        data = sitk.GetArrayFromImage(image)
        z, y, x = np.indices(data.shape)
        data[(x >= 39) & (abs(z - 36) <= 2)] = 300.
        sheet = sitk.GetImageFromArray(data)
        sheet.CopyInformation(image)
        proposal = DetectionResult(branches=deepcopy(source.branches))
        with patch("backend.refined_detection.detect_daughters", return_value=proposal):
            result = detect_refined(sheet, mask)
        self.assertEqual(result.branches, [])
        self.assertTrue(any("no_closed_lumen_sections" in c["reasons"] for c in result.diagnostics["candidates"]))


if __name__ == "__main__":
    unittest.main()
