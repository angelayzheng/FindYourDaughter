"""Geometry and interaction-state checks without opening a desktop window."""

import importlib.util
import unittest

import numpy as np

from core import Scan, ScanCase, ScanGeometry, VolumeViewOptions


@unittest.skipUnless(importlib.util.find_spec("vtk"), "Install optional VTK desktop dependency")
class VolumeViewerTest(unittest.TestCase):
    def setUp(self):
        from desktop.volume_viewer import VolumeViewer

        # Oblique, anisotropic axes with a nonzero translation.
        self.affine = np.array([[0, -2, 0.2, 10], [0.7, 0, 0, -20], [0, 0, 3, 30], [0, 0, 0, 1.]])
        self.data = np.arange(24 * 20 * 18, dtype=np.float32).reshape(24, 20, 18)
        mask = np.zeros(self.data.shape, dtype=np.float32)
        mask[3:5, 5:8, 0:16] = 1
        geometry = ScanGeometry(self.affine, "mm")
        self.case = ScanCase(Scan(self.data, geometry), Scan(mask, geometry))
        self.viewer = VolumeViewer(self.case, VolumeViewOptions(max_dimension=16))
        self.addCleanup(self.viewer.close)

    def test_numpy_vtk_order_and_downsample_spacing(self):
        image = self.viewer.volume_image
        self.assertEqual(image.GetDimensions(), (12, 10, 9))
        self.assertEqual(image.GetSpacing(), (2, 2, 2))
        self.assertEqual(image.GetScalarComponentAsDouble(2, 3, 4, 0), self.data[4, 6, 8])
        point = np.array([4, 6, 8, 1.])
        actual = self.viewer.volume.GetMatrix().MultiplyPoint(point)
        np.testing.assert_allclose(actual, self.affine @ point)

    def test_full_resolution_mask_keeps_thin_foreground_and_scan_boundary(self):
        mesh = self.viewer.mask_actor.GetMapper().GetInput()
        np.testing.assert_allclose(mesh.GetBounds(), (2.5, 4.5, 4.5, 7.5, -0.5, 15.5))
        self.assertGreater(mesh.GetNumberOfCells(), 0)
        point = np.array([3, 5, 0, 1.])
        np.testing.assert_allclose(self.viewer.mask_actor.GetMatrix().MultiplyPoint(point), self.affine @ point)

    def test_full_resolution_slice_plane_and_texture_follow_selected_index(self):
        from vtk.util.numpy_support import vtk_to_numpy

        self.viewer.set_slice(2, 7)
        plane = self.viewer.plane_sources[0]
        np.testing.assert_allclose(plane.GetOrigin(), (self.affine @ [-0.5, -0.5, 7, 1])[:3])
        texture = self.viewer.slice_actors[0][0].GetTexture().GetInput()
        self.assertEqual(texture.GetDimensions(), (24, 20, 1))
        pixels = vtk_to_numpy(texture.GetPointData().GetScalars()).reshape(20, 24, 3).transpose(1, 0, 2)
        # A masked voxel is red, while background is grayscale.
        self.assertGreater(pixels[3, 5, 0], pixels[3, 5, 1])
        self.assertEqual(pixels[0, 0, 0], pixels[0, 0, 1])
        self.assertEqual(self.viewer.sliders[2].GetRepresentation().GetValue(), 7)

    def test_controls_update_transfer_and_clip_without_mutating_source(self):
        original = self.data.copy()
        self.viewer.set_window(level=100, width=200)
        self.viewer.set_opacity(0.3)
        function = self.viewer.volume_property.GetScalarOpacity()
        self.assertEqual(function.GetValue(0), 0)
        self.assertAlmostEqual(function.GetValue(200), 0.3)
        self.viewer.clip_enabled = True
        self.viewer.set_slice(2, 5)
        self.assertTrue(self.viewer.volume_mapper.GetCropping())
        self.assertEqual(self.viewer.volume_mapper.GetCroppingRegionPlanes()[5], 5)
        np.testing.assert_array_equal(self.data, original)

    def test_empty_mask_has_no_surface(self):
        from desktop.volume_viewer import mask_surface

        self.assertIsNone(mask_surface(np.zeros((4, 5, 6))))
        self.assertIsNone(mask_surface(np.full((4, 5, 6), np.nan)))

    def test_selected_frame_is_used_with_static_mask(self):
        from desktop.volume_viewer import VolumeViewer

        image = Scan(np.stack((self.data, self.data + 100), axis=3), self.case.image.geometry)
        viewer = VolumeViewer(ScanCase(image, self.case.mask), VolumeViewOptions(frame=1))
        self.addCleanup(viewer.close)
        np.testing.assert_array_equal(viewer.data, self.data + 100)
        np.testing.assert_array_equal(viewer.mask, self.case.mask.data)


class VolumeOptionsTest(unittest.TestCase):
    def test_invalid_settings_are_rejected(self):
        for values in ({"frame": -1}, {"max_dimension": 0}, {"window": 0}, {"level": np.nan}, {"opacity": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                VolumeViewOptions(**values)


if __name__ == "__main__":
    unittest.main()
