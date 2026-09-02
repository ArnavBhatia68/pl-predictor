import unittest

import numpy as np
import pandas as pd

from pl_predictor.features import build_features, model_feature_columns
from pl_predictor.live import LiveFeatureState


def _match(date: str, home: str, away: str, hg: int, ag: int, hs: int, ass: int):
    return {
        "season_start": 2025,
        "season": "2025/26",
        "Date": date,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": hg,
        "FTAG": ag,
        "FTR": "H" if hg > ag else "D" if hg == ag else "A",
        "HS": hs,
        "AS": ass,
        "HST": max(1, hs // 3),
        "AST": max(1, ass // 3),
        "HC": 5,
        "AC": 4,
        "HF": 10,
        "AF": 11,
        "HY": 2,
        "AY": 3,
        "HR": 0,
        "AR": 0,
    }


class LiveFeatureTests(unittest.TestCase):
    def test_live_snapshot_matches_historical_pre_match_features(self) -> None:
        matches = pd.DataFrame(
            [
                _match("2025-08-01", "Chelsea", "Arsenal", 2, 0, 15, 6),
                _match("2025-08-08", "Everton", "Chelsea", 1, 1, 10, 9),
                _match("2025-08-15", "Chelsea", "Everton", 0, 1, 11, 7),
            ]
        )
        historical = build_features(matches)
        state = LiveFeatureState().replay(matches.iloc[:2])
        live = state.fixture_features("Chelsea", "Everton", "2025-08-15")
        columns = model_feature_columns(historical)
        np.testing.assert_allclose(
            live.loc[0, columns].to_numpy(dtype=float),
            historical.loc[2, columns].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_unknown_team_is_rejected(self) -> None:
        matches = pd.DataFrame(
            [_match("2025-08-01", "Chelsea", "Arsenal", 2, 0, 15, 6)]
        )
        state = LiveFeatureState().replay(matches)
        with self.assertRaises(ValueError):
            state.fixture_features("Chelsea", "Barcelona", "2025-08-08")


if __name__ == "__main__":
    unittest.main()
