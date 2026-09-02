import unittest

import numpy as np
import pandas as pd

from pl_predictor.v2 import (
    TARGET_TO_INT,
    expected_calibration_error,
    sample_weights,
    select_feature_columns,
    walk_forward_splits,
)


class V2Tests(unittest.TestCase):
    def test_time_decay_and_draw_weight(self) -> None:
        dates = pd.Series(["2020-01-01", "2024-01-01", "2024-01-01"])
        targets = pd.Series([TARGET_TO_INT["H"], TARGET_TO_INT["H"], TARGET_TO_INT["D"]])
        weights = sample_weights(dates, targets, half_life_years=2.0, draw_weight=1.10)
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[2], weights[1])
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_compact_feature_selection(self) -> None:
        frame = pd.DataFrame(
            columns=[
                "season_start", "season", "Date", "HomeTeam", "AwayTeam", "target",
                "home_elo", "away_elo", "diff_elo", "diff_last5_points",
                "diff_last5_fouls", "home_last5_matches_available",
                "away_last5_matches_available",
            ]
        )
        selected = select_feature_columns(frame, "compact")
        self.assertIn("diff_last5_points", selected)
        self.assertNotIn("diff_last5_fouls", selected)
        self.assertIn("home_last5_matches_available", selected)

    def test_target_columns_are_never_model_features(self) -> None:
        frame = pd.DataFrame(
            columns=[
                "season_start", "season", "Date", "HomeTeam", "AwayTeam",
                "target", "target_int", "home_elo", "away_elo", "diff_elo",
            ]
        )
        for feature_set in ("compact", "full"):
            selected = select_feature_columns(frame, feature_set)
            self.assertNotIn("target", selected)
            self.assertNotIn("target_int", selected)

    def test_walk_forward_has_no_future_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "season_start": [2020, 2021, 2022],
                "Date": pd.to_datetime(["2020-08-01", "2021-08-01", "2022-08-01"]),
            }
        )
        splits = walk_forward_splits(frame, (2021, 2022))
        for _, train, validation in splits:
            self.assertLess(train["Date"].max(), validation["Date"].min())

    def test_ece_is_zero_for_perfect_confidence(self) -> None:
        truth = np.array([0, 1, 2])
        probabilities = np.eye(3)
        self.assertAlmostEqual(expected_calibration_error(truth, probabilities), 0.0)


if __name__ == "__main__":
    unittest.main()
