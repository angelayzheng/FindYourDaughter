import unittest

import numpy as np

from core.denoise import denoise_volume


class DenoiseTest(unittest.TestCase):
    def test_isolated_voxel_is_replaced_but_supported_structure_remains(self):
        values = np.zeros((7, 7, 7), dtype=np.float32)
        values[3, 3, 3] = 100
        values[2:5, 2:5, 2:5] += 50
        original = values.copy()

        filtered = denoise_volume(values, tolerance=1, min_neighbors=2)

        self.assertEqual(filtered[3, 3, 3], 50)
        self.assertEqual(filtered[2, 2, 2], 50)
        np.testing.assert_array_equal(values, original)


if __name__ == "__main__":
    unittest.main()
