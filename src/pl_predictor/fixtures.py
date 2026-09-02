from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol

import httpx

FINISHED_STATUSES = {"FINISHED", "AWARDED"}
PREDICTABLE_STATUSES = {"SCHEDULED", "TIMED"}


@dataclass(frozen=True)
class Fixture:
    provider: str
    fixture_id: str
    competition: str
    season_start: int
    kickoff_utc: datetime
    status: str
    home_team: str
    away_team: str
    matchday: int | None = None
    home_goals: int | None = None
    away_goals: int | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.fixture_id}"


class FixtureProvider(Protocol):
    def fetch_matches(self, date_from: date, date_to: date) -> list[Fixture]: ...


class FootballDataOrgProvider:
    """Small adapter for football-data.org's Premier League match endpoint."""

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = "https://api.football-data.org/v4",
        competition: str = "PL",
        client: httpx.Client | None = None,
    ) -> None:
        self.token = token or os.getenv("FOOTBALL_DATA_API_TOKEN")
        if not self.token:
            raise ValueError("FOOTBALL_DATA_API_TOKEN is required to sync fixtures")
        self.base_url = base_url.rstrip("/")
        self.competition = competition
        self.client = client

    @staticmethod
    def _parse_kickoff(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(UTC)

    @staticmethod
    def _score(match: dict[str, Any], side: str) -> int | None:
        value = match.get("score", {}).get("fullTime", {}).get(side)
        return int(value) if value is not None else None

    def _fixture(self, match: dict[str, Any]) -> Fixture:
        season_start = int(str(match["season"]["startDate"])[:4])
        return Fixture(
            provider="football-data.org",
            fixture_id=str(match["id"]),
            competition=self.competition,
            season_start=season_start,
            matchday=int(match["matchday"]) if match.get("matchday") is not None else None,
            kickoff_utc=self._parse_kickoff(match["utcDate"]),
            status=str(match["status"]).upper(),
            home_team=str(match["homeTeam"]["name"]),
            away_team=str(match["awayTeam"]["name"]),
            home_goals=self._score(match, "home"),
            away_goals=self._score(match, "away"),
        )

    def fetch_matches(self, date_from: date, date_to: date) -> list[Fixture]:
        params = {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()}
        headers = {"X-Auth-Token": self.token}
        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=20.0)
        try:
            try:
                response = client.get(
                    f"{self.base_url}/competitions/{self.competition}/matches",
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise RuntimeError("Fixture provider request failed") from exc
        finally:
            if owns_client:
                client.close()
        return [self._fixture(match) for match in payload.get("matches", [])]
