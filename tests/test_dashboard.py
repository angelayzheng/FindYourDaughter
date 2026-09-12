"""Dashboard discovery and CPU 3-D preview checks."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

from frontend.cases import find_subjects, mask_choices


class CaseDiscoveryTest(unittest.TestCase):
    def test_subjects_and_neighboring_masks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "subject001"
            subject.mkdir()
            image = subject / "orig1.nii.gz"
            mask = subject / "mask1.nii"
            image.touch()
            mask.touch()
            (subject / "notes.txt").touch()
            self.assertEqual(find_subjects(root), {"subject001": [image]})
            self.assertEqual(mask_choices(image), [mask])
            self.assertEqual(find_subjects(root / "missing"), {})


class BrowserRenderTest(unittest.TestCase):
    def test_fallback_points_use_physical_affine_and_mask(self):
        import numpy as np
        from core import Scan, ScanCase, ScanGeometry
        from frontend.point_cloud import sample_scene

        data = np.full((16, 16, 16), -500, dtype=np.float32)
        data[4:12, 4:12, 4:12] = 150
        mask = np.zeros_like(data)
        mask[7:9, 7:9, 2:14] = 1
        geometry = ScanGeometry(np.array([[2, 0, 0, 10], [0, 3, 0, 20],
                                          [0, 0, 4, 30], [0, 0, 0, 1]]), "mm")
        case = ScanCase(Scan(data, geometry), Scan(mask, geometry))
        ct_points, mask_points = sample_scene(case, window=400, level=40)
        self.assertFalse(ct_points.empty)
        self.assertFalse(mask_points.empty)
        self.assertIn((24.0, 41.0, 38.0),
                      [tuple(row) for row in mask_points[["x", "y", "z"]].to_numpy()])

    def test_cpu_preview_is_visible_and_changes_with_rotation(self):
        import numpy as np
        from PIL import Image
        from core import Scan, ScanCase, ScanGeometry
        from frontend.point_cloud import render_projection

        data = np.full((16, 16, 16), -500, dtype=np.float32)
        data[4:12, 4:12, 4:12] = 150
        mask = np.zeros_like(data)
        mask[7:9, 7:9, 2:14] = 1
        case = ScanCase(Scan(data, ScanGeometry(np.eye(4), "mm")),
                        Scan(mask, ScanGeometry(np.eye(4), "mm")))
        first, ct_count, mask_count = render_projection(case, azimuth=20)
        rotated, _, _ = render_projection(case, azimuth=90)
        self.assertGreater(ct_count, 0)
        self.assertGreater(mask_count, 0)
        self.assertNotEqual(first, rotated)
        image = np.asarray(Image.open(BytesIO(first)).convert("RGB"))
        self.assertGreater(np.count_nonzero(np.any(image != [13, 18, 29], axis=2)), 1000)

    @unittest.skipUnless(sys.platform == "win32", "Bundled software OpenGL targets Windows x64")
    def test_vtk_volume_renders_offscreen_with_bundled_software_opengl(self):
        import nibabel as nib
        import numpy as np
        from PIL import Image
        from core import Scan, ScanCase, ScanGeometry, VolumeViewOptions
        from frontend.render import probe_vtk, render_3d

        supported, reason = probe_vtk()
        if not supported:
            self.skipTest(reason)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.full((16, 16, 16), -500, dtype=np.float32)
            data[4:12, 4:12, 4:12] = 150
            mask = np.zeros_like(data)
            mask[7:9, 7:9, 2:14] = 1
            nib.save(nib.Nifti1Image(data, np.eye(4)), root / "image.nii")
            nib.save(nib.Nifti1Image(mask, np.eye(4)), root / "mask.nii")
            geometry = ScanGeometry(np.eye(4), "unknown")
            case = ScanCase(Scan(data, geometry, source_path=root / "image.nii"),
                            Scan(mask, geometry, source_path=root / "mask.nii"))
            png = render_3d(case, VolumeViewOptions(max_dimension=16))
            image = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
            self.assertGreater(np.count_nonzero((image[..., 0] > 150) &
                                                (image[..., 0] > image[..., 1] * 2)), 100)

    @unittest.skipUnless(sys.platform == "win32", "Bundled software OpenGL targets Windows x64")
    def test_vtk_session_reuses_scene_and_redraws_camera(self):
        import nibabel as nib
        import numpy as np
        from core import Scan, ScanCase, ScanGeometry, VolumeViewOptions
        from frontend.render import VTKRenderSession, probe_vtk, render_3d

        supported, reason = probe_vtk()
        if not supported:
            self.skipTest(reason)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.full((24, 24, 24), -500, dtype=np.float32)
            data[2:20, 5:13, 4:19] = 150
            nib.save(nib.Nifti1Image(data, np.eye(4)), root / "image.nii")
            geometry = ScanGeometry(np.eye(4), "unknown")
            case = ScanCase(Scan(data, geometry, source_path=root / "image.nii"))
            options = VolumeViewOptions(max_dimension=24)
            session = VTKRenderSession(root / "image.nii", None, options)
            try:
                pid = session._process.pid
                first = render_3d(case, options, azimuth=0, session=session)
                second = render_3d(case, options, azimuth=90, session=session)
                slices = render_3d(case, options, show_slices=True, show_planes=True,
                                   slice_indices=(12, 12, 12), session=session)
                moved = render_3d(case, options, show_slices=True, show_planes=True,
                                  slice_indices=(12, 12, 22), session=session)
                self.assertEqual(session._process.pid, pid)
                self.assertNotEqual(first, second)
                self.assertNotEqual(slices, moved)
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
