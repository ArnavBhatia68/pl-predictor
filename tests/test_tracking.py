import math
import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

from pl_predictor.fixtures import Fixture
from pl_predictor.tracking import FixtureTracker, PredictionStore


class FakeProvider:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    def fetch_matches(self, date_from: date, date_to: date):
        return self.fixtures


class FakePredictionService:
    def __init__(self):
        self.calls = 0
        self.state = SimpleNamespace(
            last_match_date=datetime(2026, 8, 31, tzinfo=UTC)
        )
        self.model_version = "v11-standardized"

    def resolve_team(self, name):
        return name.removesuffix(" FC")

    def predict(self, home, away, fixture_date=None, season_start=None):
        self.calls += 1
        return {
            "prediction": {
                "most_likely_outcome": home,
                "most_likely_score": "2-1",
            },
            "probabilities": {"home_win": 0.6, "draw": 0.25, "away_win": 0.15},
            "expected_goals": {"home": 1.8, "away": 0.9},
            "data_as_of": "2026-08-31",
            "model_version": self.model_version,
        }


class FakeAnalytics:
    def stat_forecast(self, home, away):
        return {
            "shots": {"home": 14.0, "away": 10.0},
            "shots_on_target": {"home": 5.0, "away": 3.0},
            "corners": {"home": 6.0, "away": 4.0},
            "fouls": {"home": 10.0, "away": 12.0},
            "yellow_cards": {"home": 2.0, "away": 3.0},
        }

    def completed_match_stats(self, home, away, kickoff):
        return {
            "home_shots": 18.0,
            "away_shots": 7.0,
            "home_shots_on_target": 8.0,
            "away_shots_on_target": 2.0,
            "home_corners": 7.0,
            "away_corners": 2.0,
            "home_fouls": 9.0,
            "away_fouls": 14.0,
            "home_yellow_cards": 1.0,
            "away_yellow_cards": 4.0,
        }


def fixture(
    status="TIMED",
    home_goals=None,
    away_goals=None,
    *,
    fixture_id="1",
    kickoff=None,
):
    return Fixture(
        provider="test",
        fixture_id=fixture_id,
        competition="PL",
        season_start=2026,
        kickoff_utc=kickoff or datetime(2026, 9, 10, 18, tzinfo=UTC),
        status=status,
        home_team="Chelsea FC",
        away_team="Arsenal FC",
        matchday=4,
        home_goals=home_goals,
        away_goals=away_goals,
    )


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = PredictionStore(Path(self.temporary_directory.name) / "test.sqlite3")
        self.service = FakePredictionService()
        self.tracker = FixtureTracker(self.service, self.store)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_prediction_is_saved_once_then_graded(self):
        self.tracker.analytics = FakeAnalytics()
        before_kickoff = datetime(2026, 9, 8, 12, tzinfo=UTC)
        first = self.tracker.sync(FakeProvider([fixture()]), now=before_kickoff)
        second = self.tracker.sync(FakeProvider([fixture()]), now=before_kickoff)
        self.assertEqual(first["new_predictions"], 1)
        self.assertEqual(second["new_predictions"], 0)
        self.assertEqual(self.service.calls, 1)

        after_kickoff = datetime(2026, 9, 11, 12, tzinfo=UTC)
        completed = fixture("FINISHED", 2, 0)
        graded = self.tracker.sync(FakeProvider([completed]), now=after_kickoff)
        self.assertEqual(graded["graded"], 1)

        history = self.store.prediction_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["actual_outcome"], "home_win")
        self.assertEqual(history[0]["correct"], 1)
        self.assertAlmostEqual(history[0]["log_loss"], -math.log(0.6))
        self.assertAlmostEqual(history[0]["brier_score"], 0.245)
        self.assertEqual(history[0]["predicted_home_shots"], 14.0)
        self.assertEqual(history[0]["actual_home_shots"], 18.0)
        self.assertIn("Largest statistical misses", history[0]["review_summary"])
        self.assertEqual(self.store.record()["accuracy"], 1.0)
        self.assertEqual(self.store.team_record("Chelsea")["accuracy"], 1.0)

    def test_existing_official_prediction_gets_missing_stat_forecast_only(self):
        upcoming = fixture()
        prediction = self.service.predict("Chelsea", "Arsenal")
        prediction["prediction"]["most_likely_outcome"] = "home_win"
        self.store.upsert_fixture(upcoming, "Chelsea", "Arsenal")
        self.store.save_prediction(
            upcoming.key,
            prediction,
            created_at=datetime(2026, 9, 8, tzinfo=UTC),
        )
        original = self.store.prediction_history()[0]

        self.tracker.analytics = FakeAnalytics()
        result = self.tracker.sync(
            FakeProvider([upcoming]),
            now=datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        updated = self.store.prediction_history()[0]

        self.assertEqual(result["new_predictions"], 0)
        self.assertEqual(result["stat_forecasts_backfilled"], 1)
        self.assertEqual(updated["created_at"], original["created_at"])
        self.assertEqual(updated["home_win"], original["home_win"])
        self.assertEqual(updated["predicted_home_shots"], 14.0)

    def test_new_model_is_saved_as_shadow_without_replacing_official_pick(self):
        upcoming = fixture()
        original = self.service.predict("Chelsea", "Arsenal")
        original["prediction"]["most_likely_outcome"] = "home_win"
        original["model_version"] = "v10-official"
        self.store.upsert_fixture(upcoming, "Chelsea", "Arsenal")
        self.store.save_prediction(
            upcoming.key,
            original,
            created_at=datetime(2026, 9, 8, tzinfo=UTC),
        )

        result = self.tracker.sync(
            FakeProvider([upcoming]),
            now=datetime(2026, 9, 8, 12, tzinfo=UTC),
        )

        official = self.store.prediction_history()[0]
        shadows = self.store.shadow_prediction_history()
        self.assertEqual(result["new_shadow_predictions"], 1)
        self.assertEqual(official["model_version"], "v10-official")
        self.assertEqual(len(shadows), 1)
        self.assertEqual(shadows[0]["model_version"], "v11-standardized")

    def test_prediction_waits_for_three_day_publication_window(self):
        too_early = datetime(2026, 9, 2, 12, tzinfo=UTC)
        result = self.tracker.sync(FakeProvider([fixture()]), now=too_early)
        self.assertEqual(result["new_predictions"], 0)
        self.assertFalse(result["publication"]["is_open"])
        self.assertEqual(self.store.prediction_history(), [])

    def test_only_next_matchweek_is_published_and_early_future_pick_is_retired(self):
        gameweek_four = fixture(kickoff=datetime(2026, 9, 10, 18, tzinfo=UTC))
        gameweek_five = Fixture(
            **{
                **fixture(
                    fixture_id="2",
                    kickoff=datetime(2026, 9, 17, 18, tzinfo=UTC),
                ).__dict__,
                "matchday": 5,
            }
        )
        prediction = self.service.predict("Chelsea", "Arsenal")
        prediction["prediction"]["most_likely_outcome"] = "home_win"
        self.store.upsert_fixture(gameweek_five, "Chelsea", "Arsenal")
        self.store.save_prediction(
            gameweek_five.key,
            prediction,
            created_at=datetime(2026, 9, 2, tzinfo=UTC),
        )

        result = self.tracker.sync(
            FakeProvider([gameweek_four, gameweek_five]),
            now=datetime(2026, 9, 8, 12, tzinfo=UTC),
        )
        history = self.store.prediction_history()
        self.assertEqual(result["retired_predictions"], 1)
        self.assertEqual(result["new_predictions"], 1)
        self.assertEqual([row["fixture_key"] for row in history], [gameweek_four.key])

    def test_unknown_provider_team_is_skipped(self):
        class RejectingService(FakePredictionService):
            def resolve_team(self, name):
                raise ValueError("unknown")

        tracker = FixtureTracker(RejectingService(), self.store)
        result = tracker.sync(
            FakeProvider([fixture()]),
            now=datetime(2026, 9, 2, tzinfo=UTC),
        )
        self.assertEqual(result["saved"], 0)
        self.assertEqual(result["skipped"], 1)

    def test_new_predictions_are_blocked_when_detailed_data_is_stale(self):
        completed = fixture("FINISHED", 2, 0)
        scheduled = fixture(
            fixture_id="2",
            kickoff=datetime(2026, 9, 13, 18, tzinfo=UTC),
        )
        result = self.tracker.sync(
            FakeProvider([completed, scheduled]),
            now=datetime(2026, 9, 11, tzinfo=UTC),
        )
        self.assertTrue(result["data_stale"])
        self.assertEqual(result["prediction_blocked"], 1)
        self.assertEqual(result["new_predictions"], 0)
        self.assertEqual(self.service.calls, 0)


if __name__ == "__main__":
    unittest.main()
