"""Tests for native tensor access, spatial metadata, and export behavior."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import nibabel as nib
import numpy as np

from core import PreviewMode, PreviewOptions, Scan, ScanCase, ScanGeometry
from scripts.visualize_nifti import main


class ScanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.arange(4 * 5 * 6 * 2, dtype=np.float32).reshape(4, 5, 6, 2)
        self.geometry = ScanGeometry(np.diag([-0.7, -0.8, 1.2, 1]), "mm")
        self.scan = Scan(self.data, self.geometry)

    def test_slice_and_frame_views_preserve_native_axis_order(self) -> None:
        self.assertIs(self.scan.data, self.data)
        for axis, expected in enumerate((self.data[1, :, :, 1], self.data[:, 1, :, 1], self.data[:, :, 1, 1])):
            with self.subTest(axis=axis):
                result = self.scan.slice(axis, 1, frame=1)
                np.testing.assert_array_equal(result, expected)
                self.assertTrue(np.shares_memory(result, self.data))
        with self.assertRaises(IndexError):
            self.scan.volume(2)
        with self.assertRaises(IndexError):
            self.scan.slice(2, 6)

    def test_geometry_is_preserved_from_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.nii.gz"
            image = nib.Nifti1Image(self.data, self.geometry.affine_ras)
            image.header.set_xyzt_units("mm", "sec")
            nib.save(image, path)
            loaded = Scan.from_nifti(path)
            np.testing.assert_array_equal(loaded.data, self.data)
            np.testing.assert_allclose(loaded.geometry.affine_ras, self.geometry.affine_ras)
            np.testing.assert_allclose(loaded.geometry.spacing, (0.7, 0.8, 1.2))
            self.assertEqual(loaded.geometry.axis_codes, ("L", "P", "S"))
            self.assertEqual(loaded.geometry.spatial_unit, "mm")
            self.assertEqual(loaded.storage_dtype, "float32")
            self.assertFalse(loaded.geometry.affine_ras.flags.writeable)

    def test_case_rejects_misaligned_or_incompatible_masks(self) -> None:
        mask = Scan(np.ones(self.data.shape[:3]), self.geometry)
        self.assertIs(ScanCase(self.scan, mask).mask, mask)
        shifted = self.geometry.affine_ras.copy()
        shifted[0, 3] += 1
        for bad in (
            Scan(mask.data, ScanGeometry(shifted, "mm")),
            Scan(mask.data, ScanGeometry(self.geometry.affine_ras, "unknown")),
            Scan(np.ones((3, 5, 6)), self.geometry),
            Scan(np.ones((4, 5, 6, 3)), self.geometry),
        ):
            with self.subTest(shape=bad.shape, geometry=bad.geometry):
                with self.assertRaises(ValueError):
                    ScanCase(self.scan, bad)

    def test_numpy_export_round_trips_all_frames_and_nonfinite_voxels(self) -> None:
        self.scan.data[0, 0, 0, 0] = np.nan
        self.scan.data[0, 0, 1, 0] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            path = self.scan.export_numpy(Path(directory) / "nested" / "tensor.npy")
            restored = np.load(path, allow_pickle=False)
            np.testing.assert_array_equal(restored, self.scan.data)
            self.assertEqual(restored.dtype, np.float32)

    def test_tensor_plot_uses_actual_slice_and_patch_values(self) -> None:
        figures = []
        savefig = Figure.savefig

        def capture(figure, *args, **kwargs):
            figures.append(figure)
            return savefig(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture):
            for mode in (PreviewMode.TENSOR, PreviewMode.BOTH):
                options = PreviewOptions(mode=mode, axis=1, slice_index=2, frame=1,
                                         patch_size=2, patch_origin=(1, 3), dpi=60)
                self.scan.export_preview(Path(directory) / f"{mode.value}.png", options)
                figure = figures[-1]
                full_ax, patch_ax = figure.axes[3:5] if mode == PreviewMode.BOTH else figure.axes[:2]
                np.testing.assert_array_equal(full_ax.images[0].get_array(), self.data[:, 2, :, 1])
                expected_patch = self.data[1:3, 2, 3:5, 1]
                np.testing.assert_array_equal(patch_ax.images[0].get_array(), expected_patch)
                self.assertEqual([text.get_text() for text in patch_ax.texts],
                                 [f"{value:.6g}" for value in expected_patch.flat])

    def test_invalid_preview_settings_and_patch_bounds(self) -> None:
        for settings in ({"mode": "wrong"}, {"axis": 3}, {"frame": -1}, {"patch_size": 0},
                         {"patch_size": 17}, {"patch_origin": (-1, 0)}, {"mask_alpha": 2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PreviewOptions(**settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.png"
            with self.assertRaises(IndexError):
                self.scan.export_preview(path, PreviewOptions(mode="tensor", patch_size=2, patch_origin=(3, 4)))
            self.assertFalse(path.exists())

    def test_cli_exports_tensor_and_mask_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "subject001"
            root.mkdir()
            image = self.data[..., 0]
            mask = (image > 100).astype(np.int16)
            nib.save(nib.Nifti1Image(image, np.eye(4)), root / "orig1.nii")
            nib.save(nib.Nifti1Image(mask, np.eye(4)), root / "mask1.nii")
            output = root / "previews"
            argv = ["visualize_nifti.py", "--dataset", str(root), "--output-dir", str(output),
                    "--mode", "tensor", "--axis", "0", "--slice-index", "1", "--export-numpy"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((output / "subject001_orig1_tensor.png").exists())
            np.testing.assert_array_equal(np.load(output / "subject001_orig1.npy"), image)
            np.testing.assert_array_equal(np.load(output / "subject001_orig1_mask.npy"), mask)

    def test_core_import_does_not_load_visualization_dependencies(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import core, sys; assert 'matplotlib' not in sys.modules; assert 'streamlit' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
