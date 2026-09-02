import unittest
from datetime import date

from pl_predictor.data import _current_season_start, season_code, season_label


class DataTests(unittest.TestCase):
    def test_season_formatting(self) -> None:
        self.assertEqual(season_code(2000), "0001")
        self.assertEqual(season_code(2026), "2627")
        self.assertEqual(season_label(2000), "2000/01")

    def test_current_season_boundary(self) -> None:
        self.assertEqual(_current_season_start(date(2026, 9, 1)), 2026)
        self.assertEqual(_current_season_start(date(2027, 2, 1)), 2026)


if __name__ == "__main__":
    unittest.main()
