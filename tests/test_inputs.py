"""Regression tests for NIfTI compression, physical geometry, and recovery."""

from __future__ import annotations

import gzip
import hashlib
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from backend.inputs import load_case
from backend.pipeline import run_case
from utils.case import index_to_physical_point


class InputLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "image.nii"
        self.mask_path = self.root / "mask.nii"
        self.indices = np.indices((9, 11, 13))
        x, y, z = self.indices
        self.values = (100 + 3 * x + 5 * y - 2 * z).astype(np.float32)
        self.mask = np.zeros(self.values.shape, dtype=np.uint8)
        self.mask[2:7, 3:9, 4:10] = 1

    def write_pair(self, affine: np.ndarray, *, unit: str = "mm", compressed: bool = False) -> None:
        for path, values in ((self.image_path, self.values), (self.mask_path, self.mask)):
            image = nib.Nifti1Image(values, affine)
            image.set_sform(affine, code=2)
            image.set_qform(None, code=0)
            image.header.set_xyzt_units(unit)
            if path == self.image_path:
                image.header.set_slope_inter(2.0, -75.0)
            nib.save(image, path)
            if compressed:
                path.write_bytes(gzip.compress(path.read_bytes()))

    def digests(self) -> list[str]:
        return [hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.image_path, self.mask_path)]

    def test_native_and_mislabeled_gzip_match_simpleitk_exactly(self) -> None:
        angle = np.deg2rad(25)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([-0.7, 0.9, 1.4])
        affine[:3, 3] = (17, -31, 80)
        self.write_pair(affine)
        native = sitk.ReadImage(str(self.image_path))
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                self.write_pair(affine, compressed=compressed)
                before = self.digests()
                existing_files = set(self.root.iterdir())
                with patch("backend.inputs.tempfile.tempdir", self.temporary.name):
                    case = load_case(self.image_path, self.mask_path)
                self.assertEqual(set(self.root.iterdir()), existing_files)
                self.assertEqual(before, self.digests())
                self.assertEqual(case.image.GetSize(), native.GetSize())
                self.assertEqual(case.image.GetOrigin(), native.GetOrigin())
                self.assertEqual(case.image.GetSpacing(), native.GetSpacing())
                self.assertEqual(case.image.GetDirection(), native.GetDirection())
                np.testing.assert_array_equal(sitk.GetArrayFromImage(case.image),
                                              sitk.GetArrayFromImage(native))
                np.testing.assert_array_equal(sitk.GetArrayFromImage(case.aorta_mask), self.mask.transpose(2, 1, 0))
                for index in ((0, 0, 0), (2, 5, 7), (8, 10, 12)):
                    self.assertEqual(index_to_physical_point(case.image, index),
                                     native.TransformIndexToPhysicalPoint(index))

    def test_correctly_named_gzip_retains_native_geometry(self) -> None:
        self.write_pair(np.diag([-0.7, -0.8, 1.2, 1]), compressed=True)
        image_path, mask_path = self.root / "image.nii.gz", self.root / "mask.nii.gz"
        image_path.write_bytes(self.image_path.read_bytes())
        mask_path.write_bytes(self.mask_path.read_bytes())
        case = load_case(image_path, mask_path)
        native = sitk.ReadImage(str(image_path))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(case.image), sitk.GetArrayFromImage(native))
        self.assertEqual(case.image.GetDirection(), native.GetDirection())

    def test_sheared_pair_resamples_values_at_the_correct_physical_locations(self) -> None:
        # Include a reflection and origin offset to expose axis/sign errors.
        affine_mm = np.array([[-1.2, .18, 0, 17], [0, 1.3, -.13, -31],
                              [0, 0, 2.1, 80], [0, 0, 0, 1]], dtype=float)
        for unit, factor in (("mm", 1.0), ("meter", 1000.0), ("micron", .001), ("unknown", 1.0)):
            with self.subTest(unit=unit):
                affine = affine_mm.copy()
                affine[:3] /= factor
                self.write_pair(affine, unit=unit)
                stored = nib.load(self.image_path).affine
                lps = np.diag([-factor, -factor, factor, 1]) @ stored
                # Exercise recovery with the same misleading suffix as case 24.
                for path in (self.image_path, self.mask_path):
                    path.write_bytes(gzip.compress(path.read_bytes()))
                before = self.digests()
                with self.assertWarnsRegex(RuntimeWarning, "Resampled nonorthogonal"):
                    case = load_case(self.image_path, self.mask_path)
                self.assertEqual(self.digests(), before)
                image, mask = case.image, case.aorta_mask
                self.assertEqual(image.GetSize(), mask.GetSize())
                self.assertEqual(image.GetOrigin(), mask.GetOrigin())
                self.assertEqual(image.GetSpacing(), mask.GetSpacing())
                self.assertEqual(image.GetDirection(), mask.GetDirection())
                direction = np.array(image.GetDirection()).reshape(3, 3)
                np.testing.assert_allclose(direction.T @ direction, np.eye(3), atol=1e-12)
                np.testing.assert_array_equal(np.unique(sitk.GetArrayViewFromImage(mask)), [0, 1])
                self.assertEqual(mask.GetPixelID(), sitk.sitkUInt8)
                self.assertEqual(image.GetPixelID(), sitk.sitkFloat32)
                self.assertEqual(image.GetMetaData("branchseed_source_spatial_unit"), unit)

                checked, foreground = 0, 0
                for index in product(*[range(1, size - 1, 2) for size in image.GetSize()]):
                    point = image.TransformIndexToPhysicalPoint(index)
                    source_index = (np.linalg.inv(lps) @ [*point, 1])[:3]
                    if not np.all((source_index >= 0) & (source_index <= np.array(self.values.shape) - 1)):
                        continue
                    expected_ct = 2 * (100 + np.dot([3, 5, -2], source_index)) - 75
                    self.assertAlmostEqual(image[index], expected_ct, delta=1e-3)
                    closest = tuple(np.floor(source_index + .5).astype(int))
                    expected_mask = int(self.mask[closest])
                    self.assertEqual(mask[index], expected_mask)
                    foreground += expected_mask
                    checked += 1
                self.assertGreater(checked, 50)
                self.assertGreater(foreground, 0)

                # The orthogonal grid must cover the entire original voxel extent.
                for corner in product(*[(-.5, size - .5) for size in self.values.shape]):
                    point = (lps @ [*corner, 1])[:3]
                    output_index = np.array(image.TransformPhysicalPointToContinuousIndex(point.tolist()))
                    self.assertTrue(np.all(output_index >= -.5 - 1e-6))
                    self.assertTrue(np.all(output_index <= np.array(image.GetSize()) - .5 + 1e-6))

    def test_geometry_mismatch_is_not_hidden_by_joint_resampling(self) -> None:
        for shear in (0, .15):
            with self.subTest(shear=shear):
                affine = np.eye(4)
                affine[0, 1] = shear
                self.write_pair(affine)
                shifted = affine.copy()
                shifted[0, 3] += .25
                mask = nib.Nifti1Image(self.mask, shifted)
                mask.set_sform(shifted, code=2)
                mask.set_qform(None, code=0)
                mask.header.set_xyzt_units("mm")
                nib.save(mask, self.mask_path)
                with self.assertRaisesRegex(ValueError, "identical physical geometry"):
                    load_case(self.image_path, self.mask_path)

    def test_corrupt_gzip_input_fails_and_temporary_copy_is_removed(self) -> None:
        self.write_pair(np.eye(4))
        self.image_path.write_bytes(gzip.compress(b"not a NIfTI image"))
        before = set(self.root.iterdir())
        with patch("backend.inputs.tempfile.tempdir", self.temporary.name):
            with self.assertRaises(RuntimeError):
                load_case(self.image_path, self.mask_path)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_four_dimensional_inputs_are_rejected(self) -> None:
        data = np.zeros((4, 5, 6, 2), dtype=np.int16)
        nib.save(nib.Nifti1Image(data, np.eye(4)), self.image_path)
        nib.save(nib.Nifti1Image(data, np.eye(4)), self.mask_path)
        with self.assertRaisesRegex(ValueError, "3-D"):
            load_case(self.image_path, self.mask_path)

    def test_pipeline_accepts_small_nonorthogonality_and_mislabeled_gzip(self) -> None:
        affine = np.diag([1.5, 1.5, 1.5, 1])
        affine[1, 2] = 3.5e-4
        self.write_pair(affine, compressed=True)
        with self.assertWarnsRegex(RuntimeWarning, "Resampled nonorthogonal"):
            result = run_case(self.image_path, self.mask_path)
        self.assertEqual(result, {"case_id": "image", "parent": {"instance_id": "aorta"}, "daughters": []})


if __name__ == "__main__":
    unittest.main()
