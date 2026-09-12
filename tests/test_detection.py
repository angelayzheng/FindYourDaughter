"""Synthetic anatomy tests independent of the unannotated challenge scans."""

from __future__ import annotations

import unittest
import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk

from backend.detection import detect_daughters


def phantom(branches=(), *, spacing=(1.0, 1.0, 1.0), rotation=None, cropped=True):
    """A bright parent cylinder with optional physical-space daughter segments."""
    shape = (76, 76, 80)
    xyz = np.moveaxis(np.indices(shape)[::-1], 0, -1) * spacing
    x, y, z = np.moveaxis(xyz, -1, 0)
    parent_ct = (x - 32)**2 + (y - 32)**2 <= 5.5**2
    parent = parent_ct & (z >= 12) & (z <= 60) if cropped else parent_ct.copy()
    lumen = parent_ct.copy()
    for start, end, radius in branches:
        start, end = np.asarray(start), np.asarray(end)
        tangent = end - start
        fraction = np.clip(np.einsum("...i,i->...", xyz - start, tangent) / (tangent @ tangent), 0, 1)
        distance = np.linalg.norm(xyz - start - fraction[..., None] * tangent, axis=-1)
        lumen |= distance <= radius
    ct = ndi.gaussian_filter(np.where(lumen, 300.0, 30.0).astype(np.float32), .35)
    image, mask = sitk.GetImageFromArray(ct), sitk.GetImageFromArray(parent.astype(np.uint8))
    for volume in (image, mask):
        volume.SetSpacing(spacing)
        volume.SetOrigin((17, -31, 80))
        if rotation is not None:
            volume.SetDirection(np.asarray(rotation).ravel().tolist())
    return image, mask


class ArteryDetectionTest(unittest.TestCase):
    def test_straight_daughter_has_physical_ostium_seed_radius_and_unit_direction(self):
        image, mask = phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        result = detect_daughters(image, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        branch = result.branches[0]
        np.testing.assert_allclose(branch.ostium_xyz_mm, (54.5, 1, 116), atol=1.0)
        np.testing.assert_allclose(branch.seed_xyz_mm, (59.5, 1, 116), atol=1.0)
        self.assertAlmostEqual(np.linalg.norm(branch.direction_xyz), 1.0, places=7)
        self.assertGreater(branch.direction_xyz[0], .95)
        self.assertAlmostEqual(branch.radius_mm, 2.5, delta=1.0)
        self.assertEqual(result.daughters()[0]["instance_id"], "branch_001")
        self.assertEqual(result.daughters()[0]["parent_instance_id"], "aorta")
        length = np.linalg.norm(np.diff(branch.centerline_xyz_mm, axis=0), axis=1).sum()
        self.assertLessEqual(length, 10 + 1e-6)
        self.assertGreaterEqual(length, 5)

    def test_cropped_parent_continuations_are_not_daughters(self):
        image, mask = phantom()
        self.assertEqual(detect_daughters(image, mask).branches, [])

    def test_lateral_daughter_near_a_cropped_end_is_retained(self):
        image, mask = phantom([((32, 32, 14), (64, 32, 14), 1.7)])
        result = detect_daughters(image, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)

    def test_common_trunk_and_its_downstream_daughters_are_one_instance(self):
        image, mask = phantom([((32, 32, 36), (47, 32, 36), 2.2),
                               ((47, 32, 36), (64, 22, 36), 1.6),
                               ((47, 32, 36), (64, 42, 36), 1.6)])
        result = detect_daughters(image, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        path = np.asarray(result.branches[0].centerline_xyz_mm)
        self.assertLessEqual(np.linalg.norm(np.diff(path, axis=0), axis=1).sum(), 10 + 1e-6)

    def test_nearby_separate_ostia_remain_separate_and_ids_are_stable(self):
        image, mask = phantom([((32, 32, 30), (64, 32, 30), 1.7),
                               ((32, 32, 36), (64, 32, 36), 1.7)])
        result = detect_daughters(image, mask)
        self.assertEqual(len(result.branches), 2, result.diagnostics)
        self.assertEqual([entry["instance_id"] for entry in result.daughters()], ["branch_001", "branch_002"])
        self.assertEqual(result.daughters(), detect_daughters(image, mask).daughters())

    def test_disconnected_bright_tube_is_not_a_direct_daughter(self):
        image, mask = phantom([((44, 32, 36), (65, 32, 36), 2.0)])
        self.assertEqual(detect_daughters(image, mask).branches, [])

    def test_rotation_reflection_and_anisotropic_spacing_preserve_physical_results(self):
        direction = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        image, mask = phantom([((32, 32, 36), (61, 32, 36), 2.5)],
                               spacing=(.8, 1.0, 1.2), rotation=direction)
        result = detect_daughters(image, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        branch = result.branches[0]
        expected_root = np.array(image.GetOrigin()) + direction @ [37.5, 32, 36]
        np.testing.assert_allclose(branch.ostium_xyz_mm, expected_root, atol=1.5)
        self.assertGreater(np.dot(branch.direction_xyz, direction[:, 0]), .9)
        seed_index = image.TransformPhysicalPointToIndex(branch.seed_xyz_mm)
        self.assertEqual(mask[seed_index], 0)
        self.assertGreater(image[seed_index], 200)

    def test_empty_mask_and_uniform_unenhanced_ct_return_no_daughters(self):
        image, mask = phantom()
        self.assertEqual(detect_daughters(image, mask * 0).branches, [])
        self.assertEqual(detect_daughters(image * 0 + 30, mask).branches, [])

    def test_mismatched_geometry_is_rejected(self):
        image, mask = phantom()
        mask.SetOrigin((18, -31, 80))
        with self.assertRaisesRegex(ValueError, "geometry"):
            detect_daughters(image, mask)

    def test_seed_is_five_mm_along_a_curved_path(self):
        image, mask = phantom([((32, 32, 36), (40, 32, 36), 1.7),
                               ((40, 32, 36), (46, 39, 36), 1.7),
                               ((46, 39, 36), (62, 42, 36), 1.7)])
        result = detect_daughters(image, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        branch = result.branches[0]
        path = np.array(branch.centerline_xyz_mm)
        traveled = 0.0
        for left, right in zip(path[:-1], path[1:]):
            segment = np.linalg.norm(right - left)
            if traveled + segment >= 5:
                expected = left + (right - left) * (5 - traveled) / segment
                np.testing.assert_allclose(branch.seed_xyz_mm, expected, atol=1e-6)
                break
            traveled += segment
        else:
            self.fail("No centerline segment reaches the required seed distance")
        self.assertLess(np.linalg.norm(np.array(branch.seed_xyz_mm) - branch.ostium_xyz_mm), 4.99)


if __name__ == "__main__":
    unittest.main()
