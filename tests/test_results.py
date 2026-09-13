from pathlib import Path
import tempfile
import unittest

from frontend.results import discover_csv_files, read_csv_file, synthetic_comparison_rows


class ResultsTest(unittest.TestCase):
    def test_discovers_and_reads_nested_csv_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "synthetic" / "metrics.csv"
            csv_path.parent.mkdir()
            csv_path.write_text("case_id,f1\nsubject001,0.8\n", encoding="utf-8")

            self.assertEqual(discover_csv_files(root), [csv_path.resolve()])
            self.assertEqual(read_csv_file(csv_path), (["case_id", "f1"], [{"case_id": "subject001", "f1": "0.8"}]))

    def test_flattens_matched_missed_and_false_positive_truth(self):
        report = {
            "cases": [{
                "case_id": "subject001",
                "truth_daughters": [{"instance_id": "branch_001", "ostium_xyz_mm": [1, 2, 3]}],
                "predicted_daughters": [{"instance_id": "branch_001", "ostium_xyz_mm": [1, 2, 3]}],
                "matches": [{"truth_index": 0, "prediction_index": 0, "ostium_error_mm": 0.0, "seed_error_mm": 1.0, "radius_error_mm": 0.2}],
            }],
        }
        rows = synthetic_comparison_rows(report)
        self.assertEqual(rows[0]["status"], "matched")


if __name__ == "__main__":
    unittest.main()
