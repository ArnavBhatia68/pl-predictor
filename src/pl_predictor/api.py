from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager, suppress
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .analytics import AnalyticsService
from .data import _current_season_start, build_match_dataset
from .fixtures import FootballDataOrgProvider
from .service import PredictionService
from .tracking import FixtureTracker, PredictionStore

LOGGER = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    home_team: str = Field(min_length=2, examples=["Chelsea"])
    away_team: str = Field(min_length=2, examples=["Arsenal"])
    fixture_date: date | None = None
    season_start: int | None = Field(default=None, ge=2000, le=2100)


class FixtureSyncRequest(BaseModel):
    days_back: int = Field(default=21, ge=0, le=30)
    days_ahead: int = Field(default=28, ge=1, le=60)


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService.from_paths()


@lru_cache(maxsize=1)
def get_fixture_tracker() -> FixtureTracker:
    return FixtureTracker(
        get_prediction_service(),
        PredictionStore(),
        get_analytics_service(),
        publication_lead_days=int(os.getenv("PREDICTION_PUBLISH_LEAD_DAYS", "3")),
    )


@lru_cache(maxsize=1)
def get_analytics_service() -> AnalyticsService:
    return AnalyticsService()


def get_fixture_provider() -> FootballDataOrgProvider:
    try:
        return FootballDataOrgProvider()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def require_sync_key(
    supplied_key: Annotated[str | None, Header(alias="X-Sync-Key")] = None,
) -> None:
    expected_key = os.getenv("SYNC_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=503, detail="SYNC_API_KEY is not configured")
    if supplied_key is None or not secrets.compare_digest(supplied_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid sync key")


_refresh_lock = threading.Lock()
_refresh_trigger_lock = threading.Lock()
_refresh_status: dict[str, Any] = {
    "state": "waiting",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "result": None,
}
_players_by_team: dict[str, list[dict[str, Any]]] = {}
_player_feed_status: dict[str, Any] = {
    "available": False,
    "reason": "Player statistics have not been refreshed yet.",
}


def refresh_status() -> dict[str, Any]:
    return dict(_refresh_status)


def _run_refresh_in_background() -> None:
    try:
        result = refresh_live_data()
        LOGGER.info("Triggered live refresh completed: %s", result)
    except Exception:
        LOGGER.exception("Triggered live refresh failed")


def start_refresh_if_due(now: datetime | None = None) -> dict[str, Any]:
    """Start a bounded public refresh only when the configured interval has elapsed.

    The endpoint using this helper is safe for the scheduled workflow to call without
    exposing the administrative sync key. Repeated callers cannot force extra provider
    requests because a running refresh and a recently completed refresh are both skipped.
    """
    now = now or datetime.now(UTC)
    interval_hours = float(os.getenv("AUTO_REFRESH_INTERVAL_HOURS", "0"))
    if interval_hours <= 0:
        return {
            "started": False,
            "wait_for_completion": False,
            "reason": "automatic refresh is disabled",
            "refresh": refresh_status(),
        }

    with _refresh_trigger_lock:
        status = refresh_status()
        if status.get("state") in {"queued", "running", "waiting"}:
            return {
                "started": False,
                "wait_for_completion": True,
                "reason": "refresh is already starting or running",
                "refresh": status,
            }

        completed_at = None if status.get("state") == "error" else status.get("completed_at")
        if completed_at:
            completed = datetime.fromisoformat(str(completed_at))
            if completed.tzinfo is None:
                completed = completed.replace(tzinfo=UTC)
            due_at = completed + timedelta(hours=interval_hours)
            if now < due_at:
                return {
                    "started": False,
                    "wait_for_completion": False,
                    "reason": "refresh interval has not elapsed",
                    "due_at": due_at.isoformat(),
                    "refresh": status,
                }

        previous_completed_at = completed_at
        _refresh_status.update(state="queued", error=None)
        threading.Thread(
            target=_run_refresh_in_background,
            name="pl-predictor-live-refresh",
            daemon=True,
        ).start()
        return {
            "started": True,
            "wait_for_completion": True,
            "previous_completed_at": previous_completed_at,
            "refresh": refresh_status(),
        }


def refresh_live_data(days_back: int = 21, days_ahead: int = 28) -> dict[str, Any]:
    """Refresh the current season before syncing and grading live fixtures."""
    if not _refresh_lock.acquire(blocking=False):
        return {"skipped": True, "reason": "refresh already running", **refresh_status()}
    _refresh_status.update(
        state="running",
        started_at=datetime.now(UTC).isoformat(),
        error=None,
    )
    try:
        start_season = int(os.getenv("HISTORICAL_START_SEASON", "2000"))
        end_season = int(os.getenv("CURRENT_SEASON_START", str(_current_season_start())))
        build_match_dataset(
            start_season=start_season,
            end_season=end_season,
            refresh_current=True,
        )
        get_prediction_service().refresh_state()
        get_analytics_service().reload()
        get_fixture_tracker.cache_clear()
        provider = get_fixture_provider()
        result = get_fixture_tracker().sync(
            provider,
            days_back=days_back,
            days_ahead=days_ahead,
        )
        try:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for scorer in provider.fetch_scorers(100):
                if not scorer.get("team") or not scorer.get("name"):
                    continue
                try:
                    team = get_prediction_service().resolve_team(str(scorer["team"]))
                except ValueError:
                    continue
                grouped.setdefault(team, []).append(scorer)
            for players in grouped.values():
                players.sort(
                    key=lambda player: (player["goals"], player["assists"]),
                    reverse=True,
                )
            _players_by_team.clear()
            _players_by_team.update(grouped)
            _player_feed_status.update(
                available=bool(grouped),
                reason=None if grouped else "No current scorer records were returned.",
            )
        except RuntimeError:
            _player_feed_status.update(
                available=False,
                reason=(
                    "The connected football-data.org plan does not expose the current scorer feed."
                ),
            )
        _refresh_status.update(
            state="ready",
            completed_at=datetime.now(UTC).isoformat(),
            result=result,
        )
        return result
    except Exception as exc:
        _refresh_status.update(
            state="error",
            completed_at=datetime.now(UTC).isoformat(),
            error=str(exc),
        )
        raise
    finally:
        _refresh_lock.release()


async def _automatic_refresh_loop(interval_hours: float) -> None:
    while True:
        await asyncio.sleep(interval_hours * 3600)
        try:
            result = await asyncio.to_thread(refresh_live_data)
            LOGGER.info("Automatic live refresh completed: %s", result)
        except Exception:
            LOGGER.exception("Automatic live refresh failed")


async def _initial_fixture_sync() -> None:
    try:
        result = await asyncio.to_thread(refresh_live_data)
        LOGGER.info("Initial live refresh completed: %s", result)
    except Exception:
        LOGGER.exception("Initial fixture sync failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the model and replay historical state exactly once before accepting
    # concurrent health checks. On small instances, parallel cold loads can
    # exhaust memory before the first result reaches the shared cache.
    await asyncio.to_thread(get_prediction_service)
    interval_hours = float(os.getenv("AUTO_REFRESH_INTERVAL_HOURS", "0"))
    fixture_sync_task = asyncio.create_task(_initial_fixture_sync())
    refresh_task = (
        asyncio.create_task(_automatic_refresh_loop(interval_hours)) if interval_hours > 0 else None
    )
    try:
        yield
    finally:
        fixture_sync_task.cancel()
        with suppress(asyncio.CancelledError):
            await fixture_sync_task
        if refresh_task is not None:
            refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await refresh_task


PredictionServiceDependency = Annotated[PredictionService, Depends(get_prediction_service)]
FixtureTrackerDependency = Annotated[FixtureTracker, Depends(get_fixture_tracker)]
FixtureProviderDependency = Annotated[FootballDataOrgProvider, Depends(get_fixture_provider)]
AnalyticsServiceDependency = Annotated[AnalyticsService, Depends(get_analytics_service)]
SyncKeyDependency = Annotated[None, Depends(require_sync_key)]


app = FastAPI(
    title="PL Predictor API",
    version="0.11.0",
    description=(
        "Leak-free Premier League probabilities, upcoming fixtures, and immutable "
        "pre-match prediction tracking."
    ),
    lifespan=lifespan,
)

origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(
        ","
    )
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "PL Predictor API",
        "version": "0.11.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health(service: PredictionServiceDependency) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": service.model_version,
        "season": service.state.active_season_label,
        "data_as_of": service.state.last_match_date.date().isoformat()
        if service.state.last_match_date is not None
        else None,
        "refresh": refresh_status(),
    }


@app.get("/teams")
def teams(service: PredictionServiceDependency) -> dict[str, Any]:
    return {"season": service.state.active_season_label, "teams": service.teams}


@app.get("/teams/{team}")
def team_intelligence(
    team: str,
    service: PredictionServiceDependency,
    analytics: AnalyticsServiceDependency,
    tracker: FixtureTrackerDependency,
) -> dict[str, Any]:
    try:
        resolved = service.resolve_team(team)
        profile = analytics.team_profile(resolved)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        **profile,
        "predictions": tracker.store.team_predictions(resolved, 1000),
        "model_record": tracker.store.team_record(resolved, analytics.season_start),
        "data_as_of": service.state.last_match_date.date().isoformat()
        if service.state.last_match_date is not None
        else None,
    }


@app.post("/predict")
def predict(
    request: PredictionRequest,
    service: PredictionServiceDependency,
) -> dict[str, Any]:
    try:
        return service.predict(
            request.home_team,
            request.away_team,
            request.fixture_date,
            request.season_start,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/model-performance")
def model_performance(
    service: PredictionServiceDependency,
) -> dict[str, Any]:
    return service.performance()


@app.get("/historical-predictions")
def historical_predictions(
    service: PredictionServiceDependency,
    limit: Annotated[int, Query(ge=1, le=380)] = 20,
) -> dict[str, Any]:
    return {"predictions": service.historical_predictions(limit)}


@app.get("/fixtures/upcoming")
def upcoming_fixtures(
    tracker: FixtureTrackerDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict[str, Any]:
    return {"fixtures": tracker.store.upcoming(limit)}


@app.get("/fixtures/{fixture_key}")
def fixture_intelligence(
    fixture_key: str,
    tracker: FixtureTrackerDependency,
    analytics: AnalyticsServiceDependency,
) -> dict[str, Any]:
    fixture = tracker.store.fixture(fixture_key)
    if fixture is None:
        raise HTTPException(status_code=404, detail="Fixture not found")
    try:
        intelligence = analytics.match_center(fixture["home_team"], fixture["away_team"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    players = [
        {**player, "team": team}
        for team in (fixture["home_team"], fixture["away_team"])
        for player in _players_by_team.get(team, [])[:3]
    ]
    intelligence["players_to_watch"] = {
        **_player_feed_status,
        "players": players,
    }
    for metric in ("shots", "shots_on_target", "corners", "fouls", "yellow_cards"):
        home_value = fixture.get(f"predicted_home_{metric}")
        away_value = fixture.get(f"predicted_away_{metric}")
        if home_value is not None and away_value is not None:
            intelligence["stat_forecast"][metric] = {
                "home": home_value,
                "away": away_value,
                "total": round(float(home_value) + float(away_value), 1),
                "method": "locked with the official pre-match prediction",
            }
    return {
        "fixture": fixture,
        "shadow_predictions": tracker.store.fixture_shadows(fixture_key),
        **intelligence,
    }


@app.get("/dashboard")
def dashboard(
    service: PredictionServiceDependency,
    analytics: AnalyticsServiceDependency,
    tracker: FixtureTrackerDependency,
) -> dict[str, Any]:
    return {
        "season": analytics.season_label,
        "season_start": analytics.season_start,
        "data_as_of": service.state.last_match_date.date().isoformat()
        if service.state.last_match_date is not None
        else None,
        "matchweek": tracker.store.matchweek_board(),
        "table": analytics.table(),
        "record": tracker.store.record(analytics.season_start),
        "team_records": tracker.store.team_records(analytics.season_start),
        "publication": (refresh_status().get("result") or {}).get("publication"),
        "model": service.performance(),
        "refresh": refresh_status(),
    }


@app.get("/refresh-status")
def get_refresh_status() -> dict[str, Any]:
    return refresh_status()


@app.post("/automation/refresh-due")
def automation_refresh_due() -> dict[str, Any]:
    """Start at most one interval-limited refresh for the external scheduler."""
    return start_refresh_if_due()


@app.post("/fixtures/sync")
def sync_fixtures(
    request: FixtureSyncRequest,
    _: SyncKeyDependency,
    tracker: FixtureTrackerDependency,
    provider: FixtureProviderDependency,
) -> dict[str, Any]:
    try:
        return tracker.sync(
            provider,
            days_back=request.days_back,
            days_ahead=request.days_ahead,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/admin/refresh")
def admin_refresh(
    request: FixtureSyncRequest,
    _: SyncKeyDependency,
) -> dict[str, Any]:
    try:
        return refresh_live_data(request.days_back, request.days_ahead)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/live-predictions")
def live_predictions(
    tracker: FixtureTrackerDependency,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> dict[str, Any]:
    return {"predictions": tracker.store.prediction_history(limit)}


@app.get("/shadow-predictions")
def shadow_predictions(
    tracker: FixtureTrackerDependency,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> dict[str, Any]:
    return {"predictions": tracker.store.shadow_prediction_history(limit)}


@app.get("/season-record")
def season_record(
    tracker: FixtureTrackerDependency,
    service: PredictionServiceDependency,
    team: str | None = None,
) -> dict[str, Any]:
    if team is None:
        season_start = service.state.active_season
        return tracker.store.record(season_start)
    try:
        resolved = service.resolve_team(team)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return tracker.store.team_record(resolved, service.state.active_season)
