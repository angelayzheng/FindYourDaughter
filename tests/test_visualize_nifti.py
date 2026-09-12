"""Regression coverage for compressed volumes with misleading extensions."""

import contextlib
import gzip
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scripts.visualize_nifti import load_volume, main, nii_to_png, render_volume


class PreviewTest(unittest.TestCase):
    def test_loading_preserves_scaled_voxels_for_each_encoding(self) -> None:
        data = np.arange(120, dtype=np.int16).reshape(4, 5, 6)
        with tempfile.TemporaryDirectory() as directory:
            for image_type in (nib.Nifti1Image, nib.Nifti2Image):
                for encoding in ("plain", "gzip", "misnamed_gzip"):
                    with self.subTest(image_type=image_type.__name__, encoding=encoding):
                        image = image_type(data, np.eye(4))
                        image.header.set_slope_inter(2, -100)
                        payload = image.to_bytes()
                        if encoding != "plain":
                            payload = gzip.compress(payload)
                        path = Path(directory) / ("scan.nii.gz" if encoding == "gzip" else "scan.nii")
                        path.write_bytes(payload)
                        np.testing.assert_array_equal(load_volume(path), data * 2 - 100)

    def test_misnamed_image_and_mask_render_red_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, mask_path = root / "orig16.nii", root / "mask16.nii"
            data = np.arange(120, dtype=np.int16).reshape(4, 5, 6)
            mask = np.zeros(data.shape, dtype=np.int16)
            mask[1:3, 1:4, 2:5] = 1
            for path, voxels in ((image_path, data), (mask_path, mask)):
                path.write_bytes(gzip.compress(nib.Nifti1Image(voxels, np.eye(4)).to_bytes()))
            output = render_volume(image_path, [image_path, mask_path], root, False)
            pixels = plt.imread(output)
            red = (pixels[..., 0] > pixels[..., 1] + 0.1) & (pixels[..., 0] > pixels[..., 2] + 0.1)
            self.assertTrue(red.any())
            self.assertTrue(nii_to_png(image_path).is_file())

    def test_summary_counts_successes_and_continues_after_truncated_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.nii").write_bytes(b"\x1f\x8b")
            nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.int16), np.eye(4)), root / "good.nii")
            stdout = io.StringIO()
            with patch("sys.argv", ["visualize_nifti.py", "--dataset", str(root), "--output-dir", str(root / "previews")]):
                with contextlib.redirect_stdout(stdout):
                    main()
            self.assertIn("Skipped", stdout.getvalue())
            self.assertIn("Rendered 1 preview(s)", stdout.getvalue())
            self.assertIn("skipped 1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
