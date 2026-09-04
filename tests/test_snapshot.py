import unittest

from pl_predictor.snapshot import merge_prediction_snapshots


def prediction(key: str, *, model: str, probability: float, status: str = "TIMED"):
    return {
        "fixture_key": key,
        "kickoff_utc": "2026-09-04T19:00:00+00:00",
        "status": status,
        "created_at": "2026-09-02T20:58:55+00:00",
        "model_version": model,
        "home_win": 0.1,
        "draw": 0.2,
        "away_win": probability,
        "expected_home_goals": 0.9,
        "expected_away_goals": 2.9,
        "predicted_outcome": "away_win",
        "predicted_score": "0-2",
        "predicted_home_shots": 10.0,
        "actual_outcome": None,
        "correct": None,
        "graded_at": None,
    }


class SnapshotTests(unittest.TestCase):
    def test_empty_live_ledger_cannot_erase_durable_predictions(self):
        durable = {"predictions": [prediction("fixture:1", model="official", probability=0.7)]}
        merged = merge_prediction_snapshots(durable, {"predictions": []})
        self.assertEqual(len(merged["predictions"]), 1)
        self.assertEqual(merged["predictions"][0]["fixture_key"], "fixture:1")

    def test_official_pick_is_immutable_while_result_fields_advance(self):
        official = prediction("fixture:1", model="v4-official", probability=0.7)
        replacement = prediction(
            "fixture:1", model="v11-standardized", probability=0.8, status="FINISHED"
        )
        replacement.update(
            home_goals=0,
            away_goals=2,
            actual_outcome="away_win",
            correct=1,
            graded_at="2026-09-04T21:00:00+00:00",
            predicted_home_shots=14.0,
        )

        merged = merge_prediction_snapshots(
            {"predictions": [official]},
            {"predictions": [replacement]},
        )["predictions"][0]

        self.assertEqual(merged["model_version"], "v4-official")
        self.assertEqual(merged["away_win"], 0.7)
        self.assertEqual(merged["predicted_home_shots"], 10.0)
        self.assertEqual(merged["status"], "FINISHED")
        self.assertEqual(merged["away_goals"], 2)
        self.assertEqual(merged["correct"], 1)

    def test_new_live_prediction_is_added(self):
        merged = merge_prediction_snapshots(
            {"predictions": [prediction("fixture:1", model="v4", probability=0.7)]},
            {"predictions": [prediction("fixture:2", model="v11", probability=0.6)]},
        )
        self.assertEqual(
            {row["fixture_key"] for row in merged["predictions"]},
            {"fixture:1", "fixture:2"},
        )


if __name__ == "__main__":
    unittest.main()
