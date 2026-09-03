from __future__ import annotations

import math
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from .config import LIVE_DB_PATH
from .fixtures import FINISHED_STATUSES, PREDICTABLE_STATUSES, Fixture, FixtureProvider
from .service import PredictionService

FORECAST_METRICS = ("shots", "shots_on_target", "corners", "fouls", "yellow_cards")


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
        self._restore_remote_snapshot()

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
                    predicted_home_shots REAL,
                    predicted_away_shots REAL,
                    predicted_home_shots_on_target REAL,
                    predicted_away_shots_on_target REAL,
                    predicted_home_corners REAL,
                    predicted_away_corners REAL,
                    predicted_home_fouls REAL,
                    predicted_away_fouls REAL,
                    predicted_home_yellow_cards REAL,
                    predicted_away_yellow_cards REAL,
                    actual_outcome TEXT,
                    correct INTEGER,
                    log_loss REAL,
                    brier_score REAL,
                    graded_at TEXT,
                    actual_home_shots REAL,
                    actual_away_shots REAL,
                    actual_home_shots_on_target REAL,
                    actual_away_shots_on_target REAL,
                    actual_home_corners REAL,
                    actual_away_corners REAL,
                    actual_home_fouls REAL,
                    actual_away_fouls REAL,
                    actual_home_yellow_cards REAL,
                    actual_away_yellow_cards REAL,
                    review_summary TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff
                    ON fixtures(kickoff_utc);
                CREATE INDEX IF NOT EXISTS idx_fixtures_status
                    ON fixtures(status);
                """
            )
            existing = {
                row[1] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
            }
            additions = {
                **{
                    f"predicted_{side}_{metric}": "REAL"
                    for metric in FORECAST_METRICS
                    for side in ("home", "away")
                },
                **{
                    f"actual_{side}_{metric}": "REAL"
                    for metric in FORECAST_METRICS
                    for side in ("home", "away")
                },
                "review_summary": "TEXT",
            }
            for column, column_type in additions.items():
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE predictions ADD COLUMN {column} {column_type}"
                    )

    def _restore_remote_snapshot(self) -> None:
        """Restore the public, non-sensitive season snapshot into an empty store."""
        snapshot_url = os.getenv("PREDICTION_SEED_URL")
        if not snapshot_url:
            return
        with self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
        if count:
            return
        try:
            response = httpx.get(snapshot_url, timeout=8.0, follow_redirects=True)
            if response.status_code == 404:
                return
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("predictions", payload) if isinstance(payload, dict) else payload
            if isinstance(rows, list):
                self.import_snapshot(rows)
        except (httpx.HTTPError, ValueError, TypeError):
            # A missing snapshot must never prevent the API from starting.
            return

    def import_snapshot(self, rows: list[dict[str, Any]]) -> int:
        imported = 0
        with self._connect() as connection:
            for row in rows:
                required = {
                    "fixture_key",
                    "provider",
                    "fixture_id",
                    "competition",
                    "season_start",
                    "kickoff_utc",
                    "status",
                    "home_team",
                    "away_team",
                    "home_win",
                    "draw",
                    "away_win",
                    "expected_home_goals",
                    "expected_away_goals",
                    "predicted_outcome",
                    "predicted_score",
                }
                if not required.issubset(row):
                    continue
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fixtures (
                        fixture_key, provider, fixture_id, competition, season_start,
                        matchday, kickoff_utc, status, home_team, away_team,
                        home_goals, away_goals, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["fixture_key"],
                        row["provider"],
                        row["fixture_id"],
                        row["competition"],
                        row["season_start"],
                        row.get("matchday"),
                        row["kickoff_utc"],
                        row["status"],
                        row["home_team"],
                        row["away_team"],
                        row.get("home_goals"),
                        row.get("away_goals"),
                        row.get("updated_at") or _utc_now().isoformat(),
                    ),
                )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO predictions (
                        fixture_key, created_at, model_version, data_as_of,
                        home_win, draw, away_win, expected_home_goals, expected_away_goals,
                        predicted_outcome, predicted_score, actual_outcome, correct,
                        log_loss, brier_score, graded_at,
                        predicted_home_shots, predicted_away_shots,
                        predicted_home_shots_on_target, predicted_away_shots_on_target,
                        predicted_home_corners, predicted_away_corners,
                        predicted_home_fouls, predicted_away_fouls,
                        predicted_home_yellow_cards, predicted_away_yellow_cards,
                        actual_home_shots, actual_away_shots,
                        actual_home_shots_on_target, actual_away_shots_on_target,
                        actual_home_corners, actual_away_corners,
                        actual_home_fouls, actual_away_fouls,
                        actual_home_yellow_cards, actual_away_yellow_cards,
                        review_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["fixture_key"],
                        row.get("created_at")
                        or row.get("prediction_created_at")
                        or _utc_now().isoformat(),
                        row.get("model_version", "v4-ensemble"),
                        row.get("data_as_of", "unknown"),
                        row["home_win"],
                        row["draw"],
                        row["away_win"],
                        row["expected_home_goals"],
                        row["expected_away_goals"],
                        row["predicted_outcome"],
                        row["predicted_score"],
                        row.get("actual_outcome"),
                        row.get("correct"),
                        row.get("log_loss"),
                        row.get("brier_score"),
                        row.get("graded_at"),
                        *(
                            row.get(f"predicted_{side}_{metric}")
                            for metric in FORECAST_METRICS
                            for side in ("home", "away")
                        ),
                        *(
                            row.get(f"actual_{side}_{metric}")
                            for metric in FORECAST_METRICS
                            for side in ("home", "away")
                        ),
                        row.get("review_summary"),
                    ),
                )
                imported += int(cursor.rowcount == 1)
        return imported

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

    def save_prediction(
        self,
        fixture_key: str,
        prediction: dict[str, Any],
        stat_forecast: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> bool:
        probabilities = prediction["probabilities"]
        expected_goals = prediction["expected_goals"]
        result = prediction["prediction"]
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO predictions (
                    fixture_key, created_at, model_version, data_as_of,
                    home_win, draw, away_win, expected_home_goals, expected_away_goals,
                    predicted_outcome, predicted_score,
                    predicted_home_shots, predicted_away_shots,
                    predicted_home_shots_on_target, predicted_away_shots_on_target,
                    predicted_home_corners, predicted_away_corners,
                    predicted_home_fouls, predicted_away_fouls,
                    predicted_home_yellow_cards, predicted_away_yellow_cards
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fixture_key,
                    (created_at or _utc_now()).isoformat(),
                    prediction["model_version"],
                    prediction["data_as_of"],
                    probabilities["home_win"],
                    probabilities["draw"],
                    probabilities["away_win"],
                    expected_goals["home"],
                    expected_goals["away"],
                    result["most_likely_outcome"],
                    result["most_likely_score"],
                    *(
                        (stat_forecast or {}).get(metric, {}).get(side)
                        for metric in FORECAST_METRICS
                        for side in ("home", "away")
                    ),
                ),
            )
            return cursor.rowcount == 1

    def retire_premature_future_predictions(
        self,
        *,
        official_fixture_keys: set[str],
        publication_opens_at: datetime | None,
        now: datetime,
    ) -> int:
        """Remove unplayed predictions that were locked before the publication policy allowed."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.fixture_key, p.created_at
                FROM predictions p
                JOIN fixtures f USING (fixture_key)
                WHERE p.graded_at IS NULL
                  AND f.status IN ('SCHEDULED', 'TIMED')
                  AND f.kickoff_utc > ?
                """,
                (now.isoformat(),),
            ).fetchall()
            retired = []
            for row in rows:
                key = str(row["fixture_key"])
                created_at = datetime.fromisoformat(str(row["created_at"]))
                created_too_early = (
                    publication_opens_at is not None
                    and key in official_fixture_keys
                    and created_at < publication_opens_at
                )
                if key not in official_fixture_keys or created_too_early:
                    retired.append((key,))
            if retired:
                connection.executemany(
                    "DELETE FROM predictions WHERE fixture_key=? AND graded_at IS NULL",
                    retired,
                )
        return len(retired)

    def has_prediction(self, fixture_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM predictions WHERE fixture_key=?",
                (fixture_key,),
            ).fetchone()
        return row is not None

    @staticmethod
    def _review_summary(row: sqlite3.Row, actual_stats: dict[str, Any] | None) -> str:
        labels = {
            "home_win": str(row["home_team"]),
            "draw": "a draw",
            "away_win": str(row["away_team"]),
        }
        pick = labels[str(row["predicted_outcome"])]
        actual = labels[_outcome(int(row["home_goals"]), int(row["away_goals"]))]
        probability = {
            "home_win": float(row["home_win"]),
            "draw": float(row["draw"]),
            "away_win": float(row["away_win"]),
        }[str(row["predicted_outcome"])]
        if pick == actual:
            opening = f"The outcome call was correct: {pick} at {probability:.0%}."
        else:
            opening = f"The model favored {pick} at {probability:.0%}, but the result was {actual}."
        score = (
            f" It predicted {row['predicted_score']}; the match finished "
            f"{row['home_goals']}-{row['away_goals']}."
        )
        if not actual_stats:
            return opening + score

        misses: list[tuple[float, str]] = []
        for metric, label in (
            ("shots", "shots"),
            ("shots_on_target", "shots on target"),
            ("corners", "corners"),
            ("yellow_cards", "yellow cards"),
        ):
            for side, team_key in (("home", "home_team"), ("away", "away_team")):
                predicted = row[f"predicted_{side}_{metric}"]
                observed = actual_stats.get(f"{side}_{metric}")
                if predicted is None or observed is None:
                    continue
                text = (
                    f"{row[team_key]} {label} were {float(observed):.0f} "
                    f"versus {float(predicted):.1f} forecast"
                )
                misses.append((abs(float(observed) - float(predicted)), text))
        misses.sort(key=lambda item: item[0], reverse=True)
        detail = "; ".join(text for _, text in misses[:2])
        return opening + score + (f" Largest statistical misses: {detail}." if detail else "")

    def grade_finished(self, analytics: Any | None = None) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT f.fixture_key, f.kickoff_utc, f.home_team, f.away_team,
                       f.home_goals, f.away_goals, p.*
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
                actual_stats = (
                    analytics.completed_match_stats(
                        str(row["home_team"]),
                        str(row["away_team"]),
                        str(row["kickoff_utc"]),
                    )
                    if analytics is not None
                    else None
                )
                review = self._review_summary(row, actual_stats)
                connection.execute(
                    """
                    UPDATE predictions
                    SET actual_outcome=?, correct=?, log_loss=?, brier_score=?, graded_at=?,
                        actual_home_shots=?, actual_away_shots=?,
                        actual_home_shots_on_target=?, actual_away_shots_on_target=?,
                        actual_home_corners=?, actual_away_corners=?,
                        actual_home_fouls=?, actual_away_fouls=?,
                        actual_home_yellow_cards=?, actual_away_yellow_cards=?,
                        review_summary=?
                    WHERE fixture_key=?
                    """,
                    (
                        actual,
                        int(row["predicted_outcome"] == actual),
                        log_loss,
                        brier,
                        _utc_now().isoformat(),
                        *(
                            (actual_stats or {}).get(f"{side}_{metric}")
                            for metric in FORECAST_METRICS
                            for side in ("home", "away")
                        ),
                        review,
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

    def fixture(self, fixture_key: str) -> dict[str, Any] | None:
        rows = self._query_rows(
            """
            SELECT f.*, p.*, p.created_at AS prediction_created_at
            FROM fixtures f
            LEFT JOIN predictions p USING (fixture_key)
            WHERE f.fixture_key=?
            """,
            (fixture_key,),
        )
        return rows[0] if rows else None

    def team_predictions(self, team: str, limit: int = 20) -> list[dict[str, Any]]:
        return self._query_rows(
            """
            SELECT f.*, p.*, p.created_at AS prediction_created_at
            FROM predictions p
            JOIN fixtures f USING (fixture_key)
            WHERE f.home_team=? OR f.away_team=?
            ORDER BY f.kickoff_utc DESC
            LIMIT ?
            """,
            (team, team, limit),
        )

    def matchweek_board(self) -> dict[str, Any]:
        rows = self._query_rows(
            """
            SELECT f.*, p.*, p.created_at AS prediction_created_at
            FROM fixtures f
            LEFT JOIN predictions p USING (fixture_key)
            ORDER BY f.kickoff_utc
            """,
            (),
        )
        upcoming = [row for row in rows if row["status"] in PREDICTABLE_STATUSES]
        completed = [row for row in rows if row["status"] in FINISHED_STATUSES]
        next_matchday = (
            int(upcoming[0]["matchday"])
            if upcoming and upcoming[0]["matchday"] is not None
            else None
        )
        previous_matchday = max(
            (int(row["matchday"]) for row in completed if row["matchday"] is not None),
            default=None,
        )
        return {
            "next_matchday": next_matchday,
            "previous_matchday": previous_matchday,
            "upcoming": [
                row for row in upcoming if next_matchday is None or row["matchday"] == next_matchday
            ],
            "previous_results": [
                row
                for row in completed
                if previous_matchday is None or row["matchday"] == previous_matchday
            ],
        }

    def prediction_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._query_rows(
            """
            SELECT f.*, p.*
            FROM predictions p
            JOIN fixtures f USING (fixture_key)
            ORDER BY f.kickoff_utc DESC
            LIMIT ?
            """,
            (limit,),
        )

    def record(self, season_start: int | None = None) -> dict[str, Any]:
        where = "WHERE f.season_start=?" if season_start is not None else ""
        parameters: tuple[Any, ...] = (season_start,) if season_start is not None else ()
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS predictions,
                       COUNT(p.graded_at) AS graded,
                       AVG(CASE WHEN p.graded_at IS NOT NULL THEN p.correct END) AS accuracy,
                       AVG(p.log_loss) AS log_loss,
                       AVG(p.brier_score) AS brier_score
                FROM predictions p
                JOIN fixtures f USING (fixture_key)
                {where}
                """,
                parameters,
            ).fetchone()
        return dict(row)

    def team_record(self, team: str, season_start: int | None = None) -> dict[str, Any]:
        season_clause = "AND f.season_start=?" if season_start is not None else ""
        parameters: tuple[Any, ...] = (team, team, team, team)
        if season_start is not None:
            parameters += (season_start,)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS predictions,
                       COUNT(p.graded_at) AS graded,
                       AVG(CASE WHEN p.graded_at IS NOT NULL THEN p.correct END) AS accuracy,
                       AVG(p.log_loss) AS log_loss,
                       AVG(p.brier_score) AS brier_score,
                       AVG(CASE
                         WHEN p.graded_at IS NOT NULL AND f.home_team=?
                           THEN ABS(p.expected_home_goals - f.home_goals)
                         WHEN p.graded_at IS NOT NULL AND f.away_team=?
                           THEN ABS(p.expected_away_goals - f.away_goals)
                       END) AS goal_mae
                FROM predictions p
                JOIN fixtures f USING (fixture_key)
                WHERE (f.home_team=? OR f.away_team=?)
                {season_clause}
                """,
                parameters,
            ).fetchone()
        return {"team": team, **dict(row)}

    def team_records(self, season_start: int | None = None) -> list[dict[str, Any]]:
        season_clause = "WHERE f.season_start=?" if season_start is not None else ""
        parameters: tuple[Any, ...] = (season_start,) if season_start is not None else ()
        with self._connect() as connection:
            teams = [
                str(row[0])
                for row in connection.execute(
                    f"""
                    SELECT home_team FROM fixtures f JOIN predictions p USING (fixture_key)
                    {season_clause}
                    UNION
                    SELECT away_team FROM fixtures f JOIN predictions p USING (fixture_key)
                    {season_clause}
                    ORDER BY 1
                    """,
                    parameters + parameters,
                ).fetchall()
            ]
        return [self.team_record(team, season_start) for team in teams]

    def _query_rows(self, query: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]


class FixtureTracker:
    def __init__(
        self,
        service: PredictionService,
        store: PredictionStore,
        analytics: Any | None = None,
        publication_lead_days: int = 3,
    ) -> None:
        self.service = service
        self.store = store
        self.analytics = analytics
        self.publication_lead_days = publication_lead_days

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
        upcoming = sorted(
            (
                fixture
                for fixture in fixtures
                if fixture.status in PREDICTABLE_STATUSES and fixture.kickoff_utc > now
            ),
            key=lambda fixture: fixture.kickoff_utc,
        )
        if upcoming:
            upcoming_season = int(upcoming[0].season_start)
            active_season = getattr(self.service.state, "active_season", upcoming_season)
            if upcoming_season > active_season and hasattr(self.service, "prepare_season"):
                season_fixtures = [
                    fixture for fixture in fixtures if fixture.season_start == upcoming_season
                ]
                self.service.prepare_season(
                    upcoming_season,
                    [
                        team
                        for fixture in season_fixtures
                        for team in (fixture.home_team, fixture.away_team)
                    ],
                )
        next_matchday = upcoming[0].matchday if upcoming else None
        official_fixtures = [
            fixture
            for fixture in upcoming
            if (
                fixture.matchday == next_matchday
                if next_matchday is not None
                else fixture is upcoming[0]
            )
        ]
        first_kickoff = min(
            (fixture.kickoff_utc for fixture in official_fixtures),
            default=None,
        )
        opens_at = (
            first_kickoff - timedelta(days=self.publication_lead_days)
            if first_kickoff is not None
            else None
        )
        publication_open = opens_at is not None and now >= opens_at
        official_keys = (
            {fixture.key for fixture in official_fixtures} if publication_open else set()
        )
        retired = self.store.retire_premature_future_predictions(
            official_fixture_keys=official_keys,
            publication_opens_at=opens_at if publication_open else None,
            now=now,
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
                and fixture.key in official_keys
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
                stat_forecast = (
                    self.analytics.stat_forecast(home, away)
                    if self.analytics is not None
                    else None
                )
                predicted += int(
                    self.store.save_prediction(
                        fixture.key,
                        prediction,
                        stat_forecast,
                        created_at=now,
                    )
                )
        graded = self.store.grade_finished(self.analytics)
        return {
            "fetched": len(fixtures),
            "saved": saved,
            "new_predictions": predicted,
            "graded": graded,
            "skipped": skipped,
            "prediction_blocked": prediction_blocked,
            "retired_predictions": retired,
            "data_stale": data_stale,
            "data_as_of": data_as_of.isoformat() if data_as_of is not None else None,
            "publication": {
                "policy": "next_matchweek_three_day_window",
                "lead_days": self.publication_lead_days,
                "next_matchday": next_matchday,
                "opens_at": opens_at.isoformat() if opens_at is not None else None,
                "is_open": publication_open,
                "published_fixtures": len(official_keys) if publication_open else 0,
                "total_fixtures": len(official_fixtures),
            },
        }
