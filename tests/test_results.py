from pathlib import Path
import tempfile
import unittest

from frontend.results import discover_csv_files, read_csv_file


class ResultsTest(unittest.TestCase):
    def test_discovers_and_reads_nested_csv_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "synthetic" / "metrics.csv"
            csv_path.parent.mkdir()
            csv_path.write_text("case_id,f1\nsubject001,0.8\n", encoding="utf-8")

            self.assertEqual(discover_csv_files(root), [csv_path.resolve()])
            self.assertEqual(read_csv_file(csv_path), (["case_id", "f1"], [{"case_id": "subject001", "f1": "0.8"}]))


if __name__ == "__main__":
    unittest.main()
