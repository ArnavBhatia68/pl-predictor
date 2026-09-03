import unittest

import pandas as pd

from pl_predictor.features import RAW_STAT_COLUMNS, build_features


def match(
    date: str,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    home_shots: int,
    away_shots: int,
) -> dict[str, object]:
    result = "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"
    season_start = pd.Timestamp(date).year if pd.Timestamp(date).month >= 7 else pd.Timestamp(date).year - 1
    row: dict[str, object] = {
        "season_start": season_start,
        "season": f"{season_start}/{str(season_start + 1)[-2:]}",
        "Date": date,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": home_goals,
        "FTAG": away_goals,
        "FTR": result,
        "HS": home_shots,
        "AS": away_shots,
        "HST": max(1, home_shots // 3),
        "AST": max(1, away_shots // 3),
        "HC": 5,
        "AC": 4,
        "HF": 10,
        "AF": 11,
        "HY": 2,
        "AY": 3,
        "HR": 0,
        "AR": 0,
    }
    if not set(RAW_STAT_COLUMNS).issubset(row):
        raise AssertionError("Synthetic match is missing raw statistics")
    return row


class FeatureTests(unittest.TestCase):
    def test_current_match_statistics_never_enter_current_features(self) -> None:
        base = pd.DataFrame(
            [
                match("2025-08-01", "Chelsea", "Arsenal", 1, 0, 10, 8),
                match("2025-08-08", "Chelsea", "Liverpool", 2, 1, 12, 9),
            ]
        )
        changed = base.copy()
        changed.loc[0, ["FTHG", "FTAG", "FTR", "HS", "AS"]] = [8, 7, "H", 40, 35]

        first = build_features(base)
        second = build_features(changed)

        feature_columns = [
            column for column in first.columns if column not in {"target", "FTHG", "FTAG"}
        ]
        pd.testing.assert_series_equal(
            first.loc[0, feature_columns], second.loc[0, feature_columns], check_names=False
        )
        self.assertNotEqual(
            first.loc[1, "home_last5_goals_for"], second.loc[1, "home_last5_goals_for"]
        )
        self.assertNotEqual(
            first.loc[1, "home_last5_shots_for"], second.loc[1, "home_last5_shots_for"]
        )

    def test_rolling_features_describe_prior_matches(self) -> None:
        frame = pd.DataFrame(
            [
                match("2025-08-01", "Chelsea", "Arsenal", 2, 0, 15, 6),
                match("2025-08-08", "Liverpool", "Chelsea", 1, 1, 10, 9),
                match("2025-08-15", "Chelsea", "Everton", 0, 1, 11, 7),
            ]
        )
        features = build_features(frame)

        self.assertTrue(pd.isna(features.loc[0, "home_last5_points"]))
        self.assertAlmostEqual(features.loc[1, "away_last5_points"], 3.0)
        self.assertAlmostEqual(features.loc[2, "home_last5_points"], 2.0)
        self.assertAlmostEqual(features.loc[2, "home_last5_goals_for"], 1.5)

    def test_recent_form_crosses_season_boundary_but_season_count_resets(self) -> None:
        frame = pd.DataFrame(
            [
                match("2025-05-20", "Chelsea", "Arsenal", 2, 0, 15, 6),
                match("2025-08-15", "Chelsea", "Everton", 1, 1, 11, 7),
            ]
        )
        features = build_features(frame)

        self.assertAlmostEqual(features.loc[1, "home_last5_points"], 3.0)
        self.assertEqual(features.loc[1, "home_last5_matches_available"], 1.0)
        self.assertEqual(features.loc[1, "home_season_matches_available"], 0.0)
        self.assertAlmostEqual(features.loc[1, "home_season_goals_for"], 2.0)


if __name__ == "__main__":
    unittest.main()
