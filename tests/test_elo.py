import unittest

from pl_predictor.elo import EloRatings


class EloTests(unittest.TestCase):
    def test_elo_updates_are_zero_sum(self) -> None:
        elo = EloRatings(home_advantage=0)
        before_total = elo.get("Chelsea") + elo.get("Arsenal")
        elo.update("Chelsea", "Arsenal", 2, 0)
        after_total = elo.get("Chelsea") + elo.get("Arsenal")

        self.assertAlmostEqual(after_total, before_total)
        self.assertGreater(elo.get("Chelsea"), 1500)
        self.assertLess(elo.get("Arsenal"), 1500)

    def test_season_regression_moves_ratings_toward_mean(self) -> None:
        elo = EloRatings(home_advantage=0, season_regression=0.20)
        elo.update("Chelsea", "Arsenal", 3, 0)
        before = elo.get("Chelsea")
        elo.regress_to_mean()
        self.assertGreater(elo.get("Chelsea"), 1500)
        self.assertLess(elo.get("Chelsea"), before)


if __name__ == "__main__":
    unittest.main()
