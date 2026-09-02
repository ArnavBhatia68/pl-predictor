from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import LIVE_DB_PATH
from .fixtures import FINISHED_STATUSES, PREDICTABLE_STATUSES, Fixture, FixtureProvider
from .service import PredictionService


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals == away_goals:
        return "draw"
    return "away_win"


class PredictionStore:
    def __init__(self, path: Path = LIVE_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS fixtures (
                    fixture_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    fixture_id TEXT NOT NULL,
                    competition TEXT NOT NULL,
                    season_start INTEGER NOT NULL,
                    matchday INTEGER,
                    kickoff_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    fixture_key TEXT PRIMARY KEY REFERENCES fixtures(fixture_key),
                    created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    data_as_of TEXT NOT NULL,
                    home_win REAL NOT NULL,
                    draw REAL NOT NULL,
                    away_win REAL NOT NULL,
                    expected_home_goals REAL NOT NULL,
                    expected_away_goals REAL NOT NULL,
                    predicted_outcome TEXT NOT NULL,
                    predicted_score TEXT NOT NULL,
                    actual_outcome TEXT,
                    correct INTEGER,
                    log_loss REAL,
                    brier_score REAL,
                    graded_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff
                    ON fixtures(kickoff_utc);
                CREATE INDEX IF NOT EXISTS idx_fixtures_status
                    ON fixtures(status);
                """
            )

    def upsert_fixture(self, fixture: Fixture, home_team: str, away_team: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO fixtures (
                    fixture_key, provider, fixture_id, competition, season_start, matchday,
                    kickoff_utc, status, home_team, away_team, home_goals, away_goals, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fixture_key) DO UPDATE SET
                    season_start=excluded.season_start,
                    matchday=excluded.matchday,
                    kickoff_utc=excluded.kickoff_utc,
                    status=excluded.status,
                    home_team=excluded.home_team,
                    away_team=excluded.away_team,
                    home_goals=excluded.home_goals,
                    away_goals=excluded.away_goals,
                    updated_at=excluded.updated_at
                """,
                (
                    fixture.key,
                    fixture.provider,
                    fixture.fixture_id,
                    fixture.competition,
                    fixture.season_start,
                    fixture.matchday,
                    fixture.kickoff_utc.isoformat(),
                    fixture.status,
                    home_team,
                    away_team,
                    fixture.home_goals,
                    fixture.away_goals,
                    _utc_now().isoformat(),
                ),
            )

    def save_prediction(self, fixture_key: str, prediction: dict[str, Any]) -> bool:
        probabilities = prediction["probabilities"]
        expected_goals = prediction["expected_goals"]
        result = prediction["prediction"]
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO predictions (
                    fixture_key, created_at, model_version, data_as_of,
                    home_win, draw, away_win, expected_home_goals, expected_away_goals,
                    predicted_outcome, predicted_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fixture_key,
                    _utc_now().isoformat(),
                    prediction["model_version"],
                    prediction["data_as_of"],
                    probabilities["home_win"],
                    probabilities["draw"],
                    probabilities["away_win"],
                    expected_goals["home"],
                    expected_goals["away"],
                    result["most_likely_outcome"],
                    result["most_likely_score"],
                ),
            )
            return cursor.rowcount == 1

    def has_prediction(self, fixture_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM predictions WHERE fixture_key=?",
                (fixture_key,),
            ).fetchone()
        return row is not None

    def grade_finished(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT f.fixture_key, f.home_goals, f.away_goals,
                       p.home_win, p.draw, p.away_win, p.predicted_outcome
                FROM fixtures f
                JOIN predictions p USING (fixture_key)
                WHERE f.status IN ('FINISHED', 'AWARDED')
                  AND f.home_goals IS NOT NULL
                  AND f.away_goals IS NOT NULL
                  AND p.graded_at IS NULL
                """
            ).fetchall()
            for row in rows:
                actual = _outcome(row["home_goals"], row["away_goals"])
                probabilities = {
                    "home_win": float(row["home_win"]),
                    "draw": float(row["draw"]),
                    "away_win": float(row["away_win"]),
                }
                probability = max(probabilities[actual], 1e-15)
                log_loss = -math.log(probability)
                brier = sum(
                    (value - (1.0 if label == actual else 0.0)) ** 2
                    for label, value in probabilities.items()
                )
                connection.execute(
                    """
                    UPDATE predictions
                    SET actual_outcome=?, correct=?, log_loss=?, brier_score=?, graded_at=?
                    WHERE fixture_key=?
                    """,
                    (
                        actual,
                        int(row["predicted_outcome"] == actual),
                        log_loss,
                        brier,
                        _utc_now().isoformat(),
                        row["fixture_key"],
                    ),
                )
        return len(rows)

    def upcoming(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._query_rows(
            """
            SELECT f.*, p.created_at AS prediction_created_at,
                   p.home_win, p.draw, p.away_win,
                   p.expected_home_goals, p.expected_away_goals,
                   p.predicted_outcome, p.predicted_score
            FROM fixtures f
            LEFT JOIN predictions p USING (fixture_key)
            WHERE f.status IN ('SCHEDULED', 'TIMED')
              AND f.kickoff_utc >= ?
            ORDER BY f.kickoff_utc
            LIMIT ?
            """,
            (_utc_now().isoformat(), limit),
        )

    def prediction_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._query_rows(
            """
            SELECT f.fixture_key, f.kickoff_utc, f.status, f.home_team, f.away_team,
                   f.home_goals, f.away_goals, p.*
            FROM predictions p
            JOIN fixtures f USING (fixture_key)
            ORDER BY f.kickoff_utc DESC
            LIMIT ?
            """,
            (limit,),
        )

    def record(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS predictions,
                       COUNT(graded_at) AS graded,
                       AVG(CASE WHEN graded_at IS NOT NULL THEN correct END) AS accuracy,
                       AVG(log_loss) AS log_loss,
                       AVG(brier_score) AS brier_score
                FROM predictions
                """
            ).fetchone()
        return dict(row)

    def _query_rows(self, query: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]


class FixtureTracker:
    def __init__(self, service: PredictionService, store: PredictionStore) -> None:
        self.service = service
        self.store = store

    def sync(
        self,
        provider: FixtureProvider,
        *,
        days_back: int = 3,
        days_ahead: int = 14,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or _utc_now()
        fixtures = provider.fetch_matches(
            (now - timedelta(days=days_back)).date(),
            (now + timedelta(days=days_ahead)).date(),
        )
        latest_match_date = getattr(getattr(self.service, "state", None), "last_match_date", None)
        data_as_of = latest_match_date.date() if latest_match_date is not None else None
        newer_completed = [
            fixture
            for fixture in fixtures
            if fixture.status in FINISHED_STATUSES
            and fixture.home_goals is not None
            and fixture.away_goals is not None
            and data_as_of is not None
            and fixture.kickoff_utc.date() > data_as_of
        ]
        data_stale = bool(newer_completed)

        saved = predicted = skipped = prediction_blocked = 0
        for fixture in fixtures:
            try:
                home = self.service.resolve_team(fixture.home_team)
                away = self.service.resolve_team(fixture.away_team)
            except ValueError:
                skipped += 1
                continue
            self.store.upsert_fixture(fixture, home, away)
            saved += 1
            if (
                fixture.status in PREDICTABLE_STATUSES
                and fixture.kickoff_utc > now
                and not self.store.has_prediction(fixture.key)
            ):
                if data_stale:
                    prediction_blocked += 1
                    continue
                prediction = self.service.predict(
                    home,
                    away,
                    fixture.kickoff_utc.date(),
                    fixture.season_start,
                )
                prediction["prediction"]["most_likely_outcome"] = (
                    "home_win"
                    if prediction["prediction"]["most_likely_outcome"] == home
                    else "away_win"
                    if prediction["prediction"]["most_likely_outcome"] == away
                    else "draw"
                )
                predicted += int(self.store.save_prediction(fixture.key, prediction))
        graded = self.store.grade_finished()
        return {
            "fetched": len(fixtures),
            "saved": saved,
            "new_predictions": predicted,
            "graded": graded,
            "skipped": skipped,
            "prediction_blocked": prediction_blocked,
            "data_stale": data_stale,
            "data_as_of": data_as_of.isoformat() if data_as_of is not None else None,
        }
