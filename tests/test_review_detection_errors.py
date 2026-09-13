"""Review exports preserve scored matches and distinguish diagnostic proximity."""

import unittest

from evaluation.landmarks import score_landmarks
from scripts.review_detection_errors import review_rows
from test_evaluation import branch


class ErrorReviewTest(unittest.TestCase):
    def test_scored_assignment_is_not_replaced_by_nearest_contact(self):
        references = [branch("r1", 0.), branch("r2", 4.)]
        predictions = [branch("p1", 1.5), branch("p2", -2.5), branch("extra", 30.)]
        scores = score_landmarks(predictions, references)
        diagnostics = {"contacts": [{"ostium_xyz_mm": [0., 0., 0.], "status": "unsupported_path"}]}
        rows = review_rows("case_1", "contact", references, predictions, scores, diagnostics, 3.)
        self.assertEqual(rows[0]["prediction_id"], "p2")
        self.assertEqual(rows[1]["prediction_id"], "p1")
        self.assertEqual(rows[0]["nearest_contact_status"], "unsupported_path")
        self.assertFalse(rows[1]["contact_within_tolerance"])
        self.assertEqual(rows[2]["status"], "unmatched_prediction")
        self.assertEqual(rows[2]["nearest_reference_id"], "r2")
        self.assertNotIn("false_positive", rows[2].values())

    def test_empty_predictions_and_missing_contact_evidence_remain_explicit(self):
        references = [branch("r1", radius=None)]
        scores = score_landmarks([], references)
        rows = review_rows("case_1", "baseline", references, [], scores, {}, 3.)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "unmatched_reference")
        for key in ("prediction_id", "radius_error_mm", "nearest_contact_status",
                    "contact_within_tolerance", "nearest_prediction_distance_mm"):
            self.assertIsNone(rows[0][key])


if __name__ == "__main__":
    unittest.main()
