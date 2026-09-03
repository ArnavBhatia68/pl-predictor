import unittest
from os import environ
from typing import ClassVar
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from pl_predictor.api import (
    app,
    get_fixture_provider,
    get_fixture_tracker,
    get_prediction_service,
    require_sync_key,
)


class StubState:
    active_season_label = "2026/27"
    active_season = 2026
    last_match_date = None


class StubService:
    state = StubState()
    teams: ClassVar[list[str]] = ["Arsenal", "Chelsea"]

    def predict(self, home, away, fixture_date=None, season_start=None):
        if home == away:
            raise ValueError("Home and away teams must be different")
        return {"fixture": {"home_team": home, "away_team": away}, "probabilities": {}}

    def performance(self):
        return {"model_version": "v4-ensemble", "log_loss": 1.027664}

    def historical_predictions(self, limit=20):
        return [{"prediction": "H"}][:limit]


class StubStore:
    def upcoming(self, limit=20):
        return [{"fixture_key": "test:1"}][:limit]

    def prediction_history(self, limit=100):
        return [{"fixture_key": "test:1", "correct": 1}][:limit]

    def record(self, season_start=None):
        return {"predictions": 1, "graded": 1, "accuracy": 1.0}

    def team_record(self, team, season_start=None):
        return {"team": team, "predictions": 1, "graded": 1, "accuracy": 1.0}

    def team_records(self, season_start=None):
        return [self.team_record("Chelsea", season_start)]


class StubTracker:
    store = StubStore()

    def sync(self, provider, days_back=3, days_ahead=14):
        return {
            "fetched": 2,
            "saved": 2,
            "new_predictions": 2,
            "graded": 0,
            "skipped": 0,
            "prediction_blocked": 0,
            "data_stale": False,
        }


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app.dependency_overrides[get_prediction_service] = lambda: StubService()
        app.dependency_overrides[get_fixture_tracker] = lambda: StubTracker()
        app.dependency_overrides[get_fixture_provider] = lambda: object()
        app.dependency_overrides[require_sync_key] = lambda: None
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        app.dependency_overrides.clear()

    def test_root(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "PL Predictor API")

    def test_teams(self) -> None:
        response = self.client.get("/teams")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["teams"], ["Arsenal", "Chelsea"])

    def test_predict_validation_error(self) -> None:
        response = self.client.post(
            "/predict", json={"home_team": "Chelsea", "away_team": "Chelsea"}
        )
        self.assertEqual(response.status_code, 422)

    def test_upcoming_fixtures(self) -> None:
        response = self.client.get("/fixtures/upcoming")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["fixtures"][0]["fixture_key"], "test:1")

    def test_sync_and_record(self) -> None:
        response = self.client.post("/fixtures/sync", json={"days_ahead": 10})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["new_predictions"], 2)
        record = self.client.get("/season-record")
        self.assertEqual(record.json()["accuracy"], 1.0)

    def test_sync_key_accepts_matching_secret(self) -> None:
        with patch.dict(environ, {"SYNC_API_KEY": "test-secret"}, clear=False):
            self.assertIsNone(require_sync_key("test-secret"))

    def test_sync_key_rejects_missing_or_wrong_secret(self) -> None:
        with patch.dict(environ, {"SYNC_API_KEY": "test-secret"}, clear=False):
            with self.assertRaises(HTTPException) as missing:
                require_sync_key(None)
            self.assertEqual(missing.exception.status_code, 401)
            with self.assertRaises(HTTPException) as wrong:
                require_sync_key("wrong")
            self.assertEqual(wrong.exception.status_code, 401)

    def test_sync_key_requires_server_configuration(self) -> None:
        with patch.dict(environ, {}, clear=True):
            with self.assertRaises(HTTPException) as error:
                require_sync_key("anything")
            self.assertEqual(error.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
