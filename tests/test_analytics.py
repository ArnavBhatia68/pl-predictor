import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pl_predictor.analytics import AnalyticsService


class AnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "matches.csv"
        rows = [
            {
                "season_start": 2026,
                "season": "2026/27",
                "Date": "2026-08-10",
                "HomeTeam": "Alpha",
                "AwayTeam": "Beta",
                "FTHG": 2,
                "FTAG": 1,
                "FTR": "H",
                "HS": 12,
                "AS": 8,
                "HST": 5,
                "AST": 3,
                "HC": 6,
                "AC": 4,
                "HF": 10,
                "AF": 12,
                "HY": 2,
                "AY": 3,
                "HR": 0,
                "AR": 0,
            },
            {
                "season_start": 2026,
                "season": "2026/27",
                "Date": "2026-08-17",
                "HomeTeam": "Beta",
                "AwayTeam": "Alpha",
                "FTHG": 0,
                "FTAG": 0,
                "FTR": "D",
                "HS": 9,
                "AS": 11,
                "HST": 2,
                "AST": 4,
                "HC": 3,
                "AC": 5,
                "HF": 11,
                "AF": 9,
                "HY": 1,
                "AY": 2,
                "HR": 0,
                "AR": 0,
            },
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        self.analytics = AnalyticsService(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_table_and_recent_form(self) -> None:
        table = self.analytics.table()
        self.assertEqual(table[0]["team"], "Alpha")
        self.assertEqual(table[0]["points"], 4)
        self.assertEqual(self.analytics.team_profile("Alpha")["last_ten"]["form"], "WD")

    def test_match_center_contains_factual_and_forecast_data(self) -> None:
        center = self.analytics.match_center("Alpha", "Beta")
        self.assertEqual(center["head_to_head"]["meetings"], 2)
        self.assertGreater(center["stat_forecast"]["shots"]["total"], 0)
        self.assertFalse(center["stat_forecast"]["possession"]["available"])

    def test_live_score_overlay_updates_table_once(self) -> None:
        fixture = {
            "season_start": 2026,
            "kickoff_utc": "2026-08-24T15:00:00+00:00",
            "status": "FINISHED",
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_goals": 0,
            "away_goals": 3,
        }

        self.assertEqual(self.analytics.overlay_completed_fixtures([fixture]), 1)
        self.assertEqual(self.analytics.overlay_completed_fixtures([fixture]), 0)

        table = {row["team"]: row for row in self.analytics.table()}
        self.assertEqual(table["Alpha"]["played"], 3)
        self.assertEqual(table["Beta"]["points"], 4)
        self.assertEqual(self.analytics.recent("Beta", 1)[0]["score"], "3-0")


if __name__ == "__main__":
    unittest.main()
