"""Infrastructure smoke tests for the evaluator-facing backend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import SimpleITK as sitk

from backend.cli import main


class PipelineSmokeTest(unittest.TestCase):
    def test_cli_writes_valid_output_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "subject001"
            root.mkdir()
            image_path = root / "orig1.nii.gz"
            mask_path = root / "mask1.nii.gz"
            output_path = root / "prediction.json"

            image = sitk.Image([4, 5, 6], sitk.sitkInt16)
            image.SetSpacing((0.7, 0.7, 1.2))
            sitk.WriteImage(image, str(image_path))
            sitk.WriteImage(sitk.Image(image), str(mask_path))

            result = main(
                [
                    "--image",
                    str(image_path),
                    "--aorta-mask",
                    str(mask_path),
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(result, 0)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                {
                    "case_id": "subject001",
                    "parent": {"instance_id": "aorta"},
                    "daughters": [],
                },
            )

    def test_subject_mode_discovers_gzipped_image_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "subject042"
            root.mkdir()
            image = sitk.Image([4, 5, 6], sitk.sitkInt16)
            image.SetSpacing((0.7, 0.7, 1.2))
            sitk.WriteImage(image, str(root / "orig42.nii.gz"))
            sitk.WriteImage(sitk.Image(image), str(root / "mask42.nii.gz"))
            output_path = root / "prediction.json"

            self.assertEqual(main(["--subject", str(root), "--output", str(output_path)]), 0)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["case_id"], "subject042")
            self.assertEqual(result["parent"], {"instance_id": "aorta"})
            self.assertIsInstance(result["daughters"], list)


if __name__ == "__main__":
    unittest.main()
