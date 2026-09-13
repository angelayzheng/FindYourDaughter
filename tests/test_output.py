import unittest

from backend.output import validate_prediction


class PredictionValidationTest(unittest.TestCase):
    def test_rejects_duplicate_ids_and_non_unit_directions(self):
        daughter = {
            "instance_id": "branch_001",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [0.0, 0.0, 0.0],
            "seed_xyz_mm": [1.0, 0.0, 0.0],
            "radius_mm": 2.0,
            "direction_xyz": [2.0, 0.0, 0.0],
        }
        with self.assertRaises(ValueError):
            validate_prediction({"case_id": "case", "parent": {"instance_id": "aorta"}, "daughters": [daughter]})

        daughter["direction_xyz"] = [1.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            validate_prediction({"case_id": "case", "parent": {"instance_id": "aorta"}, "daughters": [daughter, daughter.copy()]})


if __name__ == "__main__":
    unittest.main()
