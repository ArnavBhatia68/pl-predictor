import unittest

import pandas as pd

from pl_predictor.v3 import goal_feature_columns


class V3Tests(unittest.TestCase):
    def test_goal_targets_are_never_features(self) -> None:
        frame = pd.DataFrame(
            columns=[
                "home_elo", "away_elo", "diff_elo",
                "home_last5_goals_for", "away_last5_goals_for", "diff_last5_goals_for",
                "target", "target_int", "target_home_goals", "target_away_goals", "FTHG", "FTAG",
            ]
        )
        selected = goal_feature_columns(frame)
        for target in (
            "target", "target_int", "target_home_goals", "target_away_goals", "FTHG", "FTAG"
        ):
            self.assertNotIn(target, selected)


if __name__ == "__main__":
    unittest.main()
