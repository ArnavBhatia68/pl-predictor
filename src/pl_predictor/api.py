from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager, suppress
from datetime import date
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
    days_back: int = Field(default=3, ge=0, le=30)
    days_ahead: int = Field(default=14, ge=1, le=60)


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService.from_paths()


@lru_cache(maxsize=1)
def get_fixture_tracker() -> FixtureTracker:
    return FixtureTracker(get_prediction_service(), PredictionStore())


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


def refresh_live_data(days_back: int = 3, days_ahead: int = 14) -> dict[str, Any]:
    """Refresh the current season before syncing and grading live fixtures."""
    start_season = int(os.getenv("HISTORICAL_START_SEASON", "2000"))
    end_season = int(os.getenv("CURRENT_SEASON_START", str(_current_season_start())))
    build_match_dataset(
        start_season=start_season,
        end_season=end_season,
        refresh_current=True,
    )
    get_prediction_service().refresh_state()
    get_fixture_tracker.cache_clear()
    return get_fixture_tracker().sync(
        get_fixture_provider(),
        days_back=days_back,
        days_ahead=days_ahead,
    )


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
        result = await asyncio.to_thread(
            get_fixture_tracker().sync,
            get_fixture_provider(),
        )
        LOGGER.info("Initial fixture sync completed: %s", result)
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
        asyncio.create_task(_automatic_refresh_loop(interval_hours))
        if interval_hours > 0
        else None
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
SyncKeyDependency = Annotated[None, Depends(require_sync_key)]


app = FastAPI(
    title="PL Predictor API",
    version="0.8.0",
    description=(
        "Leak-free Premier League probabilities, upcoming fixtures, and immutable "
        "pre-match prediction tracking."
    ),
    lifespan=lifespan,
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
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
        "version": "0.8.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health(service: PredictionServiceDependency) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "v4-ensemble",
        "season": service.state.active_season_label,
        "data_as_of": service.state.last_match_date.date().isoformat()
        if service.state.last_match_date is not None
        else None,
    }


@app.get("/teams")
def teams(service: PredictionServiceDependency) -> dict[str, Any]:
    return {"season": service.state.active_season_label, "teams": service.teams}


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


@app.get("/season-record")
def season_record(
    tracker: FixtureTrackerDependency,
) -> dict[str, Any]:
    return tracker.store.record()
