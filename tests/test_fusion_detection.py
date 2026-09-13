"""Independent geometry fixtures and candidate-fusion behavior, before data tuning."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import test_detection as anatomy
from backend.candidates import Candidate, group_candidates
from backend.detection import DetectedBranch
from backend.fusion_detection import FusionOptions, detect_fusion


def candidate(source, number=1, offset=(0., 0., 0.)):
    path = np.array([[d, 0., 0.] for d in range(11)]) + offset
    branch = DetectedBranch(path[0].tolist(), path[5].tolist(), 1., [1., 0., 0.],
                            path.tolist(), .5, "orthogonal_section")
    return Candidate(f"{source}_{number:03d}", source, branch)


class FusionGeometryTest(anatomy.ArteryDetectionTest):
    def setUp(self):
        replacement = patch.object(anatomy, "detect_daughters", detect_fusion)
        replacement.start()
        self.addCleanup(replacement.stop)


class FusionCandidateTest(unittest.TestCase):
    def test_duplicate_sources_are_one_group_without_averaging_their_paths(self):
        left, right = candidate("baseline"), candidate("contact", offset=(.2, .1, 0))
        before = deepcopy(left.branch)
        groups = group_candidates([right, left], ostium_mm=2.5, path_mm=1.5)
        self.assertEqual(len(groups), 1)
        self.assertEqual({c.source for c in groups[0]}, {"baseline", "contact"})
        self.assertEqual(left.branch, before)

    def test_nearby_separate_parallel_ostia_are_preserved(self):
        candidates = [candidate(source, i, (0, y, 0)) for source in ("baseline", "contact")
                      for i, y in ((1, 0), (2, 2.))]
        groups = group_candidates(candidates, ostium_mm=2.5, path_mm=1.5)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(group) == 2 for group in groups))
        self.assertTrue(all(group[0].branch.ostium_xyz_mm == group[1].branch.ostium_xyz_mm for group in groups))

    def test_two_traces_inside_one_broad_lumen_are_grouped(self):
        left, right = candidate("baseline"), candidate("contact", offset=(0, 2., 0))
        left.branch.radius_mm = right.branch.radius_mm = 3.
        self.assertEqual(len(group_candidates([left, right], ostium_mm=2.5, path_mm=1.5)), 1)

    def test_matching_does_not_transitively_merge_two_roots_from_one_source(self):
        groups = group_candidates([candidate("baseline", 1), candidate("baseline", 2, (0, 2, 0)),
                                   candidate("contact", 1, (0, 1, 0))], ostium_mm=2.5, path_mm=1.5)
        self.assertEqual(sorted(len(g) for g in groups), [1, 2])

    def test_different_outward_paths_at_nearby_origins_are_not_duplicates(self):
        left, right = candidate("baseline"), candidate("contact")
        right.branch.centerline_xyz_mm = [[0., float(d), 0.] for d in range(11)]
        right.branch.seed_xyz_mm = [0., 5., 0.]
        right.branch.direction_xyz = [0., 1., 0.]
        self.assertEqual(len(group_candidates([left, right], ostium_mm=2.5, path_mm=1.5)), 2)

    def test_low_contrast_noisy_branch_is_retained_and_inputs_are_unchanged(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        data = sitk.GetArrayFromImage(image)
        parent = sitk.GetArrayFromImage(mask) > 0
        outside = (data > 60) & ~parent
        data[outside] = 30 + (data[outside] - 30) * 120 / 270
        data += np.random.default_rng(812).normal(0, 2, data.shape).astype(np.float32)
        noisy = sitk.GetImageFromArray(data)
        noisy.CopyInformation(image)
        result = detect_fusion(noisy, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        np.testing.assert_allclose(result.branches[0].ostium_xyz_mm, (54.5, 1, 116), atol=1)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(noisy), data)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(mask) > 0, parent)
        json.dumps(result.diagnostics, allow_nan=False)

    def test_early_common_trunk_is_retained_for_review_without_child_substitution(self):
        image, mask = anatomy.phantom([((32, 32, 36), (40, 32, 36), 1.7),
                                      ((40, 32, 36), (60, 18, 36), 1.4),
                                      ((40, 32, 36), (60, 46, 36), 1.4)])
        result = detect_fusion(image, mask)
        self.assertEqual(result.branches, [])
        deferred = result.diagnostics["deferred_common_trunks"]
        self.assertEqual(len(deferred), 1)
        self.assertLess(deferred[0]["observed_length_mm"], 5)
        self.assertEqual(deferred[0]["status"], "needs_seed_policy")

    def test_radius_evidence_and_source_provenance_are_outside_daughter_schema(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        result = detect_fusion(image, mask)
        self.assertEqual(len(result.branches), 1)
        records = result.diagnostics["candidates"]
        chosen = [r for r in records if r["status"] == "accepted"]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["instance_id"], "branch_001")
        self.assertEqual(chosen[0]["features"]["radius_quality"], "stable_sections")
        self.assertEqual(len(chosen[0]["features"]["sections"]), 3)
        self.assertTrue(any(r["status"] == "duplicate" for r in records))
        self.assertEqual(len(result.daughters()[0]), 6)

    def test_options_validate_finite_physical_distances(self):
        for values in ({"duplicate_ostium_mm": np.nan}, {"duplicate_path_mm": 0},
                       {"max_radius_cv": -1}, {"minimum_support_fraction": 1.1}):
            with self.assertRaises(ValueError):
                FusionOptions(**values)

    def test_bright_pool_on_a_thin_bridge_is_not_a_daughter(self):
        image, mask = anatomy.phantom([((32, 32, 36), (43, 32, 36), 1.)])
        data = sitk.GetArrayFromImage(image)
        z, y, x = np.indices(data.shape)
        data[(x - 53)**2 + (y - 32)**2 + (z - 36)**2 <= 12**2] = 300
        image_with_pool = sitk.GetImageFromArray(data)
        image_with_pool.CopyInformation(image)
        self.assertEqual(detect_fusion(image_with_pool, mask).branches, [])

    def test_coarse_through_plane_spacing_preserves_physical_coordinates(self):
        image, mask = anatomy.phantom([((32, 32, 36), (61, 32, 36), 2.5)], spacing=(.8, 1., 3.))
        result = detect_fusion(image, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        np.testing.assert_allclose(result.branches[0].ostium_xyz_mm, (54.5, 1, 116), atol=1.5)
        self.assertAlmostEqual(np.linalg.norm(result.branches[0].direction_xyz), 1)


if __name__ == "__main__":
    unittest.main()
