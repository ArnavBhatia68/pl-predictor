import unittest

import numpy as np
import pandas as pd

from pl_predictor.stat_models import DetailedStatModels


class _Regressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


class _Pair:
    def __init__(self, home: float, away: float) -> None:
        self.home_model = _Regressor(home)
        self.away_model = _Regressor(away)


class DetailedStatModelTests(unittest.TestCase):
    def test_each_stat_is_served_and_sot_never_exceeds_shots(self) -> None:
        models = {
            "shots": _Pair(10.0, 9.0),
            "shots_on_target": _Pair(12.0, 11.0),
            "corners": _Pair(5.0, 4.0),
            "fouls": _Pair(11.0, 12.0),
            "yellow_cards": _Pair(2.0, 3.0),
        }
        forecasts = DetailedStatModels(
            models,  # type: ignore[arg-type]
            ["feature"],
            {metric: "independent-test-model" for metric in models},
            {},
        ).predict(pd.DataFrame({"feature": [1.0]}))

        self.assertEqual(set(models), set(forecasts) - {"possession"})
        self.assertEqual(forecasts["shots_on_target"]["home"], 10.0)
        self.assertEqual(forecasts["shots_on_target"]["away"], 9.0)
        self.assertEqual(len(forecasts["shots"]["home_interval_80"]), 2)
        self.assertFalse(forecasts["possession"]["available"])


if __name__ == "__main__":
    unittest.main()
