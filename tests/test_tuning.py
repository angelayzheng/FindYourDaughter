"""Search records, ranking, regression guard, and overwrite protection."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.configuration import configuration
from evaluation.tuning import GUARD, RANKING, expand_plan, ranked_trials, run_sweep, trial_id, validate_selection


def plan():
    return {"matching_tolerance_mm": 3., "shortlist_per_detector": 1, "ranking": RANKING, "synthetic_guard": GUARD,
            "families": [{"detector": "baseline", "grid": {"baseline.intensity_fraction": [.35, .55, .55]}}]}


def summary(f1=.5, recall=.5):
    return {"reference_f1": f1, "reference_recall": recall, "reference_precision": f1,
            "matched": 1, "reference_count": 2, "prediction_count": 2, "unmatched_predictions": 1,
            "unmatched_references": 1, "successful_cases": 1, "failed_cases": 0}


class TuningTest(unittest.TestCase):
    def test_grid_is_deterministic_deduplicated_and_always_includes_default(self):
        configs = expand_plan(plan())
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0], configuration("baseline"))
        self.assertEqual(configs, expand_plan(plan()))
        self.assertNotEqual(trial_id(configs[0]), trial_id(configs[1]))
        invalid = plan()
        invalid["families"][0]["grid"] = {"baseline.typo": [1]}
        with self.assertRaises(ValueError):
            expand_plan(invalid)

    def test_failed_trials_cannot_win_and_ties_prefer_fewer_changes(self):
        trials = [{"trial_id": "default", "detector": "baseline", "status": "complete",
                   "summary": summary(), "changed_parameters": {}},
                  {"trial_id": "changed", "detector": "baseline", "status": "complete",
                   "summary": summary(), "changed_parameters": {"x": 1}},
                  {"trial_id": "failed", "detector": "baseline", "status": "incomplete",
                   "summary": summary(.99, .99), "changed_parameters": {}}]
        self.assertEqual([t["trial_id"] for t in ranked_trials(trials, "baseline")], ["default", "changed"])

    def test_search_keeps_draft_winner_but_rejects_synthetic_regression_and_never_overwrites(self):
        def draft(dataset, output, *, detectors, tolerance_mm, parameters):
            tuned = parameters["baseline"]["baseline"]["intensity_fraction"] == .35
            return {"status": "complete", "backend_sha256": {}, "cases": [
                {"case_id": "case_1", "status": "ok", "input_sha256": {"case_1/orig.nii": "original"},
                 "detectors": {"baseline": {"scores": {"summary": summary()}}}}],
                "summary": {"baseline": summary(.9 if tuned else .5)}}

        def synthetic(dataset, tolerance, *, detector, parameters):
            tuned = parameters["baseline"]["intensity_fraction"] == .35
            return {"true_positive": 1 if tuned else 2, "false_positive": 0,
                    "false_negative": 1 if tuned else 0, "mean_errors": {}}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "experiment"
            with patch("evaluation.tuning._synthetic_inventory", return_value={"subject1/orig.nii": "development"}), \
                    patch("evaluation.tuning.evaluate_dataset", side_effect=draft), \
                    patch("evaluation.tuning.evaluate_synthetic", side_effect=synthetic):
                result = run_sweep(plan(), root / "draft", output, synthetic_development=root / "synthetic")
                selected = result["selection"]["baseline"]
                self.assertEqual(result["status"], "complete")
                self.assertEqual(selected["selected_trial"], trial_id(configuration("baseline")))
                self.assertNotEqual(selected["selected_trial"], selected["best_draft_trial"])
                self.assertEqual(len(list((output / "trials").glob("*/config.json"))), 2)
                before = (output / "index.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    run_sweep(plan(), root / "draft", output, synthetic_development=root / "synthetic")
                self.assertEqual(before, (output / "index.json").read_bytes())
                with self.assertRaisesRegex(ValueError, "overlap"):
                    validate_selection(output, root / "synthetic")


if __name__ == "__main__":
    unittest.main()
