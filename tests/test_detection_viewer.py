"""Detector display alignment, selection, and slice visibility regressions."""

import importlib.util
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import unittest

import numpy as np
import SimpleITK as sitk

from backend.detection import DetectedBranch, DetectionResult
from backend.inputs import CaseData
from core import Scan, ScanCase, ScanGeometry, VolumeViewOptions


@unittest.skipUnless(importlib.util.find_spec("vtk"), "Install optional VTK desktop dependency")
class DetectionViewerTest(unittest.TestCase):
    def make_viewer(self, *, empty=False):
        from desktop.volume_viewer import VolumeViewer

        affine = np.array([[0, -2, .2, 10], [.7, 0, 0, -20], [0, 0, 3, 30], [0, 0, 0, 1.]])
        geometry = ScanGeometry(affine, "mm")
        data = np.zeros((24, 20, 18), dtype=np.float32)
        mask = data.copy()
        mask[2:4, 3:9, 1:16] = 1
        case = ScanCase(Scan(data, geometry), Scan(mask, geometry))
        branches = []
        for z in (5, 12):
            path = np.array([(affine @ [x, y, z, 1])[:3] * [-1, -1, 1]
                             for x, y in ((3, 5), (5, 5), (8, 5), (9, 6))])
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            seed = np.array([np.interp(5, arc, path[:, axis]) for axis in range(3)])
            direction = seed - path[0]
            direction /= np.linalg.norm(direction)
            branches.append(DetectedBranch(path[0].tolist(), seed.tolist(), 1.4, direction.tolist(),
                                           path.tolist(), .75, "orthogonal_section"))
        result = DetectionResult([] if empty else branches)
        viewer = VolumeViewer(case, VolumeViewOptions(max_dimension=16), detection=result)
        self.addCleanup(viewer.close)
        return viewer

    def test_backend_working_grid_converts_values_and_physical_points(self):
        from desktop.detection_overlay import scan_case_from_backend

        data = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
        image = sitk.GetImageFromArray(data)
        image.SetSpacing((.7, 1.2, 2.5))
        image.SetOrigin((32, -18, 90))
        image.SetDirection((0, -1, 0, -1, 0, 0, 0, 0, 1))
        mask = sitk.Cast(image > 100, sitk.sitkUInt8)
        case = scan_case_from_backend(CaseData(Path("orig.nii"), Path("mask.nii"), image, mask))
        np.testing.assert_array_equal(case.image.data, data.transpose(2, 1, 0))
        self.assertEqual(case.image.geometry.spatial_unit, "mm")
        self.assertTrue(case.image.geometry.matches(case.mask.geometry))
        for index in ((0, 0, 0), (3, 4, 5)):
            ras = (case.image.geometry.affine_ras @ [*index, 1])[:3]
            expected = np.array(image.TransformIndexToPhysicalPoint(index)) * [-1, -1, 1]
            np.testing.assert_allclose(ras, expected)
        # The display owns its array after the backend SimpleITK objects are freed.
        del image, mask
        self.assertEqual(case.image.data[3, 4, 5], data[5, 4, 3])

    def test_overlay_converts_lps_once_and_ring_preserves_radius(self):
        viewer = self.make_viewer()
        overlay = viewer.detection_overlay
        branch = overlay.result.branches[0]
        expected = np.array(branch.seed_xyz_mm) * [-1, -1, 1]
        np.testing.assert_allclose(overlay.seeds[0], expected)
        np.testing.assert_allclose(overlay.actors[0][2].GetCenter(), expected, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(overlay.rings[0] - expected, axis=1), branch.radius_mm)
        np.testing.assert_allclose(overlay.voxels[0][0], [3, 5, 5])
        self.assertFalse(viewer.volume.GetVisibility())

    def test_selection_keys_link_slices_and_toggle_every_overlay(self):
        viewer = self.make_viewer()
        overlay = viewer.detection_overlay
        viewer.interactor.SetKeySym("Right")
        viewer._key()
        self.assertEqual(overlay.selected, 1)
        expected_voxel = np.linalg.solve(viewer.affine, np.r_[overlay.seeds[1], 1])[:3]
        np.testing.assert_array_equal(viewer.indices, np.round(expected_voxel).astype(int))
        self.assertIn("branch_002", overlay.detail.GetInput())
        self.assertTrue(any(actor.GetVisibility() for actor in overlay.slice_actors[1][0]))
        viewer.interactor.SetKeySym("d")
        viewer._key()
        self.assertFalse(overlay.visible)
        self.assertFalse(overlay.label.GetVisibility())
        self.assertTrue(all(not actor.GetVisibility() for actors in overlay.actors for actor in actors))
        self.assertTrue(all(not actor.GetVisibility() for branch in overlay.slice_actors
                            for plane in branch for actor in plane))
        viewer._key()
        self.assertTrue(overlay.visible)
        overlay.select(2)
        self.assertEqual(overlay.selected, 0)

    def test_slice_overlay_clips_paths_and_hides_distant_markers(self):
        from desktop.detection_overlay import slice_segments

        segments = slice_segments(np.array([[0., 0, -2], [4, 0, 2]]), 2, 0)
        np.testing.assert_allclose(segments, [[[1.5, 0, 0], [2.5, 0, 0]]])
        viewer = self.make_viewer()
        viewer.set_slice(2, 17)
        for branch in viewer.detection_overlay.slice_actors:
            self.assertTrue(all(not actor.GetVisibility() for actor in branch[0]))
            self.assertEqual(branch[0][0].GetMapper().GetInput().GetNumberOfPoints(), 0)

    def test_contrast_changes_preserve_candidate_focus_and_overview_resets_it(self):
        viewer = self.make_viewer()
        overlay = viewer.detection_overlay
        overlay.select(1)
        before = np.array([renderer.GetActiveCamera().GetFocalPoint() for renderer in viewer.slice_renderers])
        viewer.set_window(level=180, width=500)
        after = np.array([renderer.GetActiveCamera().GetFocalPoint() for renderer in viewer.slice_renderers])
        np.testing.assert_allclose(before, after)
        overlay.overview()
        self.assertIsNone(viewer.slice_focus_voxel)
        self.assertFalse(np.allclose(before, [r.GetActiveCamera().GetFocalPoint() for r in viewer.slice_renderers]))

    def test_clicking_a_rendered_candidate_selects_it(self):
        viewer = self.make_viewer()
        viewer.window.SetOffScreenRendering(True)
        overlay = viewer.detection_overlay
        overlay.overview()
        viewer.window.Render()
        viewer.scene.SetWorldPoint(*overlay.seeds[1], 1)
        viewer.scene.WorldToDisplay()
        x, y, _ = viewer.scene.GetDisplayPoint()
        self.assertTrue(overlay.pick(round(x), round(y)))
        self.assertEqual(overlay.selected, 1)

    def test_empty_detection_is_explicit_and_navigation_is_safe(self):
        viewer = self.make_viewer(empty=True)
        overlay = viewer.detection_overlay
        self.assertIn("No candidates", overlay.detail.GetInput())
        overlay.select(1)
        overlay.toggle()
        overlay.overview()
        self.assertIsNone(overlay.selected)
        self.assertFalse(overlay.label.GetVisibility())


class DetectionViewerCliTest(unittest.TestCase):
    def test_incompatible_detection_flags_fail_before_loading(self):
        from scripts.view_nifti_3d import main

        for flags in (("--detect", "--no-mask"), ("--detect", "--frame", "1"),
                      ("--branch", "1"), ("--detect", "--branch", "0"), ("--detector", "contact")):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main(["--image", "unused.nii", *flags])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
