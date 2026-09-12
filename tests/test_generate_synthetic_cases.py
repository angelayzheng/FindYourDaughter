import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from scripts.generate_synthetic_cases import generate_case, write_case


class SyntheticCaseTest(unittest.TestCase):
    def test_generated_pair_has_matching_grid_and_parent_only_mask(self):
        case = generate_case(
            1,
            seed=10,
            daughters=2,
            size_xyz=(64, 64, 96),
            spacing_xyz_mm=(1.0, 1.0, 1.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_case(case, root)
            image = sitk.ReadImage(str(root / "subject001" / "orig1.nii"))
            mask = sitk.ReadImage(str(root / "subject001" / "mask1.nii"))
            self.assertEqual(image.GetSize(), mask.GetSize())
            self.assertEqual(image.GetSpacing(), mask.GetSpacing())
            truth = json.loads((root / "subject001" / "truth1.json").read_text())
            self.assertEqual(len(truth["daughters"]), 2)
            centerline = np.asarray(truth["daughters"][0]["centerline_xyz_mm"])
            straight_line = centerline[0] + np.linspace(0, 1, len(centerline))[:, None] * (centerline[-1] - centerline[0])
            self.assertGreater(float(np.max(np.linalg.norm(centerline - straight_line, axis=1))), 0.01)
            for daughter in truth["daughters"]:
                self.assertAlmostEqual(np.linalg.norm(daughter["direction_xyz"]), 1.0, places=5)
                self.assertAlmostEqual(
                    np.linalg.norm(np.asarray(daughter["seed_xyz_mm"]) - daughter["ostium_xyz_mm"]),
                    5.0,
                    delta=0.2,
                )
            mask_array = sitk.GetArrayFromImage(mask)
            image_array = sitk.GetArrayFromImage(image)
            self.assertTrue(np.all(mask_array[mask_array > 0] == 1))
            self.assertGreater(float(image_array[mask_array > 0].mean()), 0)


if __name__ == "__main__":
    unittest.main()
