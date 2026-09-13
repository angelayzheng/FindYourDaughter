"""Paper-inspired detector: shared anatomy contract and targeted failure cases."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import test_detection as anatomy
from backend.cli import main
from backend.contact_detection import (
    ContactOptions, _contact_radius, _fit_axis, _supported_wall, _trace_contact, detect_contacts,
)
from backend.detection import detect_daughters
from backend.detectors import detect


class ContactGeometryTest(anatomy.ArteryDetectionTest):
    """Run the existing physical/anatomical contract against the second algorithm."""

    def setUp(self):
        replacement = patch.object(anatomy, "detect_daughters", detect_contacts)
        replacement.start()
        self.addCleanup(replacement.stop)


class ContactDetectionTest(unittest.TestCase):
    @staticmethod
    def replace_image(image, data):
        result = sitk.GetImageFromArray(data.astype(np.float32))
        result.CopyInformation(image)
        return result

    def test_weak_branch_is_recovered_without_changing_the_baseline(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        data = sitk.GetArrayFromImage(image)
        parent = sitk.GetArrayFromImage(mask) > 0
        daughter = (data > 60) & ~parent
        data[daughter] = 30 + (data[daughter] - 30) * (150 - 30) / 270
        image = self.replace_image(image, data)
        self.assertEqual(detect_daughters(image, mask).branches, [])
        result = detect_contacts(image, mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        branch = result.branches[0]
        np.testing.assert_allclose(branch.ostium_xyz_mm, (54.5, 1, 116), atol=1)
        self.assertAlmostEqual(branch.radius_mm, 2.5, delta=.6)
        self.assertGreater(branch.direction_xyz[0], .95)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(mask) > 0, parent)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), data)
        json.dumps(result.diagnostics, allow_nan=False)

    def test_thin_wall_rind_does_not_remove_a_real_opening(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        data = sitk.GetArrayFromImage(image)
        z, y, x = np.indices(data.shape)
        rind = ((x - 32)**2 + (y - 32)**2 <= 6.5**2) & (z >= 22) & (z <= 49)
        data[rind] = 300
        result = detect_contacts(self.replace_image(image, data), mask)
        self.assertEqual(len(result.branches), 1, result.diagnostics)
        self.assertGreater(result.diagnostics["removed_voxels"], 0)
        np.testing.assert_allclose(result.branches[0].ostium_xyz_mm, (54.5, 1, 116), atol=1)

    def test_air_in_section_background_does_not_merge_soft_tissue_into_radius(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 4.0)])
        data = sitk.GetArrayFromImage(image)
        _, y, _ = np.indices(data.shape)
        # A soft-tissue strip crosses the section; most of the outer ring is air.
        # An unbounded midpoint-to-air threshold would join the strip to the tube.
        background = np.where(np.abs(y - 32) < 6, 35., -800.)
        data = np.where(data > 100, data, background)
        radius, method = _contact_radius(image, data, sitk.GetArrayFromImage(mask) > 0,
                                          np.zeros(3), np.array([59.5, 1., 116.]),
                                          np.array([1., 0., 0.]), 40., 500., 6.)
        self.assertIsNotNone(radius)
        self.assertAlmostEqual(radius, 4., delta=.6)
        self.assertEqual(method, "orthogonal_section")

    def test_curved_parent_continuations_are_not_daughters(self):
        image, _ = anatomy.phantom()
        xyz = np.moveaxis(np.indices((76, 76, 80))[::-1], 0, -1)

        def tube(points):
            result = np.zeros(xyz.shape[:-1], bool)
            for left, right in zip(points[:-1], points[1:]):
                left, right = np.asarray(left), np.asarray(right)
                vector = right - left
                along = np.clip(np.einsum("...i,i->...", xyz - left, vector) / (vector @ vector), 0, 1)
                result |= np.linalg.norm(xyz - left - along[..., None] * vector, axis=-1) <= 5.5
            return result

        points = [(32, 32, 12), (32, 32, 36), (36, 32, 45), (45, 32, 50), (60, 32, 50)]
        parent = tube(points)
        lumen = tube([(32, 32, 0), *points, (79, 32, 50)])
        mask = sitk.GetImageFromArray(parent.astype(np.uint8))
        mask.CopyInformation(image)
        result = detect_contacts(self.replace_image(image, np.where(lumen, 300, 30)), mask)
        self.assertEqual(result.branches, [], result.diagnostics)
        self.assertEqual(result.diagnostics["rejected"].get("parent_end_cap"), 2)

    def test_layer_cleanup_requires_outer_support_and_keeps_it(self):
        # A thin wall band attached to one outward tube. Unequal physical spacing
        # must not change which side is the parent-facing side.
        distance = np.broadcast_to(np.arange(1, 9)[None, None, :] * .7, (7, 7, 8))
        vessels = np.zeros(distance.shape, bool)
        vessels[:, :, 0] = True
        vessels[3, 3, :] = True
        result = _supported_wall(vessels, distance, np.array([1.5, 1, .7]), 2)
        self.assertTrue(result[3, 3].all())
        self.assertFalse(result[0, 0, 0])
        np.testing.assert_array_equal(result[distance > 2], vessels[distance > 2])

    def test_early_common_trunk_is_not_split_into_two_daughters(self):
        image, mask = anatomy.phantom([((32, 32, 36), (40, 32, 36), 1.7),
                                      ((40, 32, 36), (60, 18, 36), 1.4),
                                      ((40, 32, 36), (60, 46, 36), 1.4)])
        result = detect_contacts(image, mask)
        self.assertEqual(result.branches, [])
        self.assertEqual(result.diagnostics["rejected"].get("early_bifurcation"), 1)

    def test_tracing_reports_other_contacts_and_cycles(self):
        points = np.array([[x, 0., 0.] for x in range(5)])
        chain = [{1}, {0, 2}, {1, 3}, {2, 4}, {3}]
        path, stop = _trace_contact(0, np.array([-.5, 0, 0]), points, chain, {4})
        self.assertEqual(stop, "other_contact")
        self.assertTrue(np.all(path[:, 0] < 4))
        # A cycle of sub-2-mm arms is pruned as unsupported rather than followed
        # indefinitely; a sustained loop produces an ambiguous fork or cycle.
        loop = [{1, 3}, {0, 2}, {1, 3}, {2, 0}]
        _, stop = _trace_contact(0, np.array([-.5, 0, 0]), points[:4], loop, set())
        self.assertIn(stop, ("bifurcation", "cycle"))

    def test_fitted_direction_is_physical_unit_length_and_oriented(self):
        anchor = np.array([100., -30., 400.])
        forward = np.array([1., 2., -3.])
        points = anchor + np.linspace(-2, 2, 9)[:, None] * forward
        axis = _fit_axis(points, anchor, forward)
        np.testing.assert_allclose(axis, forward / np.linalg.norm(forward), atol=1e-8)
        with self.assertRaises(ValueError):
            _fit_axis(points, anchor, np.zeros(3))

    def test_rejects_invalid_inputs_and_options(self):
        image, mask = anatomy.phantom()
        for bad in (mask * 2, sitk.Cast(mask, sitk.sitkFloat32) * .5):
            with self.assertRaisesRegex(ValueError, "binary"):
                detect_contacts(image, bad)
        data = sitk.GetArrayFromImage(image)
        data[36, 32, 32] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            detect_contacts(self.replace_image(image, data), mask)
        for kwargs in ({"tube_seed": np.nan}, {"growth_mm": 30}, {"support_band_mm": 0},
                       {"tube_seed": .01}, {"min_radius_mm": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContactOptions(**kwargs)

    def test_default_selection_preserves_refined_and_unknown_names_fail(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        expected = detect(image, mask, detector="refined").daughters()
        self.assertEqual(detect(image, mask).daughters(), expected)
        self.assertEqual(detect(image, mask, detector="baseline").daughters(), detect_daughters(image, mask).daughters())
        with self.assertRaisesRegex(ValueError, "Unknown detector"):
            detect(image, mask, detector="missing")

    def test_cli_contact_has_the_same_output_schema(self):
        image, mask = anatomy.phantom([((32, 32, 36), (64, 32, 36), 2.5)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "subject101"
            root.mkdir()
            sitk.WriteImage(image, str(root / "ct.nii.gz"))
            sitk.WriteImage(mask, str(root / "mask.nii.gz"))
            output = root / "prediction.json"
            code = main(["--image", str(root / "ct.nii.gz"), "--aorta-mask", str(root / "mask.nii.gz"),
                         "--output", str(output), "--detector", "contact"])
            document = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(set(document), {"case_id", "parent", "daughters"})
        self.assertEqual(len(document["daughters"]), 1)
        self.assertEqual(set(document["daughters"][0]), {"instance_id", "parent_instance_id", "ostium_xyz_mm",
                                                        "seed_xyz_mm", "radius_mm", "direction_xyz"})
        json.dumps(document, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
