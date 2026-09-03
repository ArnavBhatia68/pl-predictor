from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import MATCHES_PATH
from .elo import EloRatings
from .features import (
    TeamState,
    _add_differences,
    _add_summary,
    _team_match_record,
    summarize,
)


@dataclass
class LiveFeatureState:
    """Chronological team state used to build features for an unplayed fixture."""

    def __post_init__(self) -> None:
        self.elo = EloRatings()
        self.team_states: defaultdict[str, TeamState] = defaultdict(TeamState)
        self.active_season: int | None = None
        self.active_season_label: str | None = None
        self.last_match_date: pd.Timestamp | None = None
        self._active_teams: set[str] = set()

    @property
    def available_teams(self) -> list[str]:
        return sorted(self._active_teams)

    def _begin_season(self, season_start: int, season_label: str) -> None:
        if self.active_season is not None:
            if season_start < self.active_season:
                raise ValueError("Cannot move live feature state backward to an earlier season")
            if season_start > self.active_season:
                self.elo.regress_to_mean()
        self.team_states = defaultdict(TeamState)
        self._active_teams = set()
        self.active_season = season_start
        self.active_season_label = season_label

    def _ensure_season(self, season_start: int, season_label: str) -> None:
        if self.active_season != season_start:
            self._begin_season(season_start, season_label)

    def prepare_season(self, season_start: int, teams: list[str]) -> None:
        """Prepare an upcoming season before its first completed result exists."""
        if self.active_season is not None and season_start < self.active_season:
            raise ValueError("Cannot prepare an earlier season")
        if self.active_season != season_start:
            self._begin_season(
                season_start,
                f"{season_start:04d}/{(season_start + 1) % 100:02d}",
            )
        self._active_teams.update(teams)

    def _pre_match_features(
        self,
        home_team: str,
        away_team: str,
        fixture_date: pd.Timestamp,
        season_start: int,
        season_label: str,
    ) -> dict[str, object]:
        home_state = self.team_states[home_team]
        away_state = self.team_states[away_team]
        features: dict[str, object] = {
            "season_start": season_start,
            "season": season_label,
            "Date": fixture_date,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "home_elo": self.elo.get(home_team),
            "away_elo": self.elo.get(away_team),
        }
        _add_summary(features, "home_last5", summarize(list(home_state.recent)[-5:]))
        _add_summary(features, "away_last5", summarize(list(away_state.recent)[-5:]))
        _add_summary(features, "home_last10", summarize(home_state.recent))
        _add_summary(features, "away_last10", summarize(away_state.recent))
        _add_summary(features, "home_venue_last5", summarize(home_state.home))
        _add_summary(features, "away_venue_last5", summarize(away_state.away))
        _add_summary(features, "home_season", summarize(home_state.season))
        _add_summary(features, "away_season", summarize(away_state.season))
        _add_differences(features)
        return features

    def process_match(self, row: pd.Series) -> dict[str, object]:
        season_start = int(row["season_start"])
        season_label = str(row["season"])
        fixture_date = pd.Timestamp(row["Date"])
        self._ensure_season(season_start, season_label)
        home_team = str(row["HomeTeam"])
        away_team = str(row["AwayTeam"])
        features = self._pre_match_features(
            home_team, away_team, fixture_date, season_start, season_label
        )

        home_record = _team_match_record(row, is_home=True)
        away_record = _team_match_record(row, is_home=False)
        home_state = self.team_states[home_team]
        away_state = self.team_states[away_team]
        home_state.recent.append(home_record)
        away_state.recent.append(away_record)
        home_state.home.append(home_record)
        away_state.away.append(away_record)
        home_state.season.append(home_record)
        away_state.season.append(away_record)
        self.elo.update(home_team, away_team, int(row["FTHG"]), int(row["FTAG"]))
        self._active_teams.update((home_team, away_team))
        self.last_match_date = fixture_date
        return features

    def replay(self, matches: pd.DataFrame) -> LiveFeatureState:
        ordered = matches.copy()
        ordered["Date"] = pd.to_datetime(ordered["Date"], errors="raise")
        ordered = ordered.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable")
        for _, row in ordered.iterrows():
            self.process_match(row)
        return self

    def fixture_features(
        self,
        home_team: str,
        away_team: str,
        fixture_date: str | pd.Timestamp,
        season_start: int | None = None,
    ) -> pd.DataFrame:
        if self.active_season is None or self.active_season_label is None:
            raise ValueError("Live feature state has not replayed any matches")
        season_start = season_start if season_start is not None else self.active_season
        if season_start != self.active_season:
            raise ValueError("Live predictions currently support only the active season")
        if home_team == away_team:
            raise ValueError("Home and away teams must be different")
        unknown = sorted({home_team, away_team} - self._active_teams)
        if unknown:
            raise ValueError(f"Unknown active-season team(s): {', '.join(unknown)}")
        timestamp = pd.Timestamp(fixture_date)
        if self.last_match_date is not None and timestamp < self.last_match_date:
            raise ValueError(
                f"Fixture date {timestamp.date()} predates latest loaded match "
                f"{self.last_match_date.date()}"
            )
        features = self._pre_match_features(
            home_team,
            away_team,
            timestamp,
            self.active_season,
            self.active_season_label,
        )
        return pd.DataFrame([features])

    @classmethod
    def from_csv(cls, matches_path: Path = MATCHES_PATH) -> LiveFeatureState:
        return cls().replay(pd.read_csv(matches_path))
