from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MATCHES_PATH

STAT_KEYS = {
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
    "corners": ("HC", "AC"),
    "fouls": ("HF", "AF"),
    "yellow_cards": ("HY", "AY"),
    "red_cards": ("HR", "AR"),
}


def _result(goals_for: int, goals_against: int) -> str:
    return "W" if goals_for > goals_against else "D" if goals_for == goals_against else "L"


def _mean(values: pd.Series, default: float = 0.0) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else default


def _round(value: float, digits: int = 1) -> float:
    return round(float(value), digits)


@dataclass
class AnalyticsService:
    """Read-only football intelligence derived from completed match data."""

    matches_path: Path = MATCHES_PATH

    def __post_init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        matches = pd.read_csv(self.matches_path)
        matches["Date"] = pd.to_datetime(matches["Date"], errors="raise")
        self.matches = matches.sort_values(
            ["Date", "HomeTeam", "AwayTeam"], kind="stable"
        ).reset_index(drop=True)
        latest_season = int(self.matches["season_start"].max())
        self.current = self.matches[self.matches["season_start"] == latest_season].copy()
        self.season_start = latest_season
        self.season_label = str(self.current["season"].iloc[-1])

    def overlay_completed_fixtures(self, fixtures: list[dict[str, Any]]) -> int:
        """Add provider-confirmed results that the detailed statistics feed has not published.

        The fixture provider usually publishes the final score first. These score-only rows keep
        standings, recent form, and team records current while the richer match row catches up.
        Repeated refreshes are idempotent, and a real detailed row always wins over an overlay.
        """
        additions: list[dict[str, Any]] = []
        for fixture in fixtures:
            if int(fixture.get("season_start", -1)) != self.season_start:
                continue
            if fixture.get("status") not in {"FINISHED", "AWARDED"}:
                continue
            if fixture.get("home_goals") is None or fixture.get("away_goals") is None:
                continue

            kickoff = pd.Timestamp(str(fixture["kickoff_utc"])).tz_localize(None).normalize()
            home_team = str(fixture["home_team"])
            away_team = str(fixture["away_team"])
            existing = self.current[
                (self.current["HomeTeam"] == home_team)
                & (self.current["AwayTeam"] == away_team)
                & ((self.current["Date"].dt.normalize() - kickoff).abs() <= pd.Timedelta(days=1))
            ]
            if not existing.empty:
                continue

            home_goals = int(fixture["home_goals"])
            away_goals = int(fixture["away_goals"])
            row = {column: np.nan for column in self.matches.columns}
            row.update(
                {
                    "season_start": self.season_start,
                    "season": self.season_label,
                    "Date": kickoff,
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,
                    "FTHG": home_goals,
                    "FTAG": away_goals,
                    "FTR": "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A",
                }
            )
            additions.append(row)

        if not additions:
            return 0
        added = pd.DataFrame(additions, columns=self.matches.columns)
        self.matches = (
            pd.concat([self.matches, added], ignore_index=True)
            .sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable")
            .reset_index(drop=True)
        )
        self.current = self.matches[self.matches["season_start"] == self.season_start].copy()
        return len(additions)

    @property
    def teams(self) -> list[str]:
        return sorted(set(self.current["HomeTeam"]) | set(self.current["AwayTeam"]))

    def _team_matches(self, team: str, *, all_seasons: bool = False) -> pd.DataFrame:
        source = self.matches if all_seasons else self.current
        return source[(source["HomeTeam"] == team) | (source["AwayTeam"] == team)].copy()

    @staticmethod
    def _team_row(row: pd.Series, team: str) -> dict[str, Any]:
        is_home = row["HomeTeam"] == team
        own, opponent = ("H", "A") if is_home else ("A", "H")
        goals_for = int(row[f"FT{own}G"])
        goals_against = int(row[f"FT{opponent}G"])
        stats: dict[str, Any] = {}
        for label, (home_column, away_column) in STAT_KEYS.items():
            own_column = home_column if is_home else away_column
            opponent_column = away_column if is_home else home_column
            own_value = row.get(own_column)
            opponent_value = row.get(opponent_column)
            stats[label] = None if pd.isna(own_value) else float(own_value)
            stats[f"{label}_against"] = None if pd.isna(opponent_value) else float(opponent_value)
        return {
            "date": pd.Timestamp(row["Date"]).date().isoformat(),
            "season": str(row["season"]),
            "venue": "home" if is_home else "away",
            "opponent": str(row["AwayTeam"] if is_home else row["HomeTeam"]),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "score": f"{goals_for}-{goals_against}",
            "result": _result(goals_for, goals_against),
            "points": 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0,
            **stats,
        }

    def recent(self, team: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._team_matches(team).tail(limit)
        return [self._team_row(row, team) for _, row in rows.iloc[::-1].iterrows()]

    def _season_summary(self, team: str) -> dict[str, Any]:
        recent = [self._team_row(row, team) for _, row in self._team_matches(team).iterrows()]
        played = len(recent)
        points = sum(row["points"] for row in recent)
        summary: dict[str, Any] = {
            "played": played,
            "wins": sum(row["result"] == "W" for row in recent),
            "draws": sum(row["result"] == "D" for row in recent),
            "losses": sum(row["result"] == "L" for row in recent),
            "goals_for": sum(row["goals_for"] for row in recent),
            "goals_against": sum(row["goals_against"] for row in recent),
            "goal_difference": sum(row["goals_for"] - row["goals_against"] for row in recent),
            "points": points,
            "points_per_game": _round(points / played, 2) if played else 0.0,
            "form": "".join(row["result"] for row in recent[-5:]),
        }
        for label in STAT_KEYS:
            values = [row[label] for row in recent if row[label] is not None]
            allowed = [
                row[f"{label}_against"] for row in recent if row[f"{label}_against"] is not None
            ]
            summary[f"{label}_per_game"] = _round(np.mean(values)) if values else None
            summary[f"{label}_against_per_game"] = _round(np.mean(allowed)) if allowed else None
        return summary

    def table(self) -> list[dict[str, Any]]:
        rows = [{"team": team, **self._season_summary(team)} for team in self.teams]
        rows.sort(
            key=lambda row: (
                row["points"],
                row["goal_difference"],
                row["goals_for"],
            ),
            reverse=True,
        )
        for position, row in enumerate(rows, start=1):
            row["position"] = position
        return rows

    def team_profile(self, team: str) -> dict[str, Any]:
        if team not in self.teams:
            raise ValueError(f"Unknown active-season team '{team}'")
        table_row = next(row for row in self.table() if row["team"] == team)
        recent = self.recent(team, 10)
        last_ten_points = sum(row["points"] for row in recent)
        return {
            "team": team,
            "season": self.season_label,
            "table": table_row,
            "recent": recent,
            "last_ten": {
                "played": len(recent),
                "points": last_ten_points,
                "points_per_game": _round(last_ten_points / len(recent), 2) if recent else 0.0,
                "goals_for_per_game": _round(np.mean([row["goals_for"] for row in recent]))
                if recent
                else 0.0,
                "goals_against_per_game": _round(np.mean([row["goals_against"] for row in recent]))
                if recent
                else 0.0,
                "form": "".join(row["result"] for row in reversed(recent)),
            },
        }

    def head_to_head(self, team_a: str, team_b: str, limit: int = 10) -> dict[str, Any]:
        rows = self.matches[
            ((self.matches["HomeTeam"] == team_a) & (self.matches["AwayTeam"] == team_b))
            | ((self.matches["HomeTeam"] == team_b) & (self.matches["AwayTeam"] == team_a))
        ].tail(limit)
        matches: list[dict[str, Any]] = []
        wins_a = wins_b = draws = 0
        for _, row in rows.iloc[::-1].iterrows():
            home_goals, away_goals = int(row["FTHG"]), int(row["FTAG"])
            winner = (
                str(row["HomeTeam"])
                if home_goals > away_goals
                else str(row["AwayTeam"])
                if away_goals > home_goals
                else "Draw"
            )
            wins_a += int(winner == team_a)
            wins_b += int(winner == team_b)
            draws += int(winner == "Draw")
            matches.append(
                {
                    "date": pd.Timestamp(row["Date"]).date().isoformat(),
                    "season": str(row["season"]),
                    "home_team": str(row["HomeTeam"]),
                    "away_team": str(row["AwayTeam"]),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "winner": winner,
                }
            )
        return {
            "team_a": team_a,
            "team_b": team_b,
            "meetings": len(matches),
            "team_a_wins": wins_a,
            "draws": draws,
            "team_b_wins": wins_b,
            "matches": matches,
        }

    def _rolling_average(
        self,
        team: str,
        label: str,
        *,
        against: bool = False,
        venue: str | None = None,
    ) -> float:
        rows = self._team_matches(team)
        if venue == "home":
            rows = rows[rows["HomeTeam"] == team]
        elif venue == "away":
            rows = rows[rows["AwayTeam"] == team]
        rows = rows.tail(10)
        values = [
            self._team_row(row, team)[f"{label}{'_against' if against else ''}"]
            for _, row in rows.iterrows()
        ]
        clean = [float(value) for value in values if value is not None]
        return float(np.mean(clean)) if clean else 0.0

    def stat_forecast(self, home_team: str, away_team: str) -> dict[str, Any]:
        forecasts: dict[str, Any] = {}
        for label in ("shots", "shots_on_target", "corners", "fouls", "yellow_cards"):
            home_for = self._rolling_average(home_team, label)
            away_allowed = self._rolling_average(away_team, label, against=True)
            home_venue = self._rolling_average(home_team, label, venue="home")
            away_for = self._rolling_average(away_team, label)
            home_allowed = self._rolling_average(home_team, label, against=True)
            away_venue = self._rolling_average(away_team, label, venue="away")
            home_value = 0.45 * home_for + 0.35 * away_allowed + 0.20 * home_venue
            away_value = 0.45 * away_for + 0.35 * home_allowed + 0.20 * away_venue
            forecasts[label] = {
                "home": _round(home_value),
                "away": _round(away_value),
                "total": _round(home_value + away_value),
                "method": "recent attack blended with opponent defence and venue form",
            }

        forecasts["possession"] = {
            "available": False,
            "reason": "Historical possession labels are not present in the training source.",
        }
        return forecasts

    def completed_match_stats(
        self,
        home_team: str,
        away_team: str,
        kickoff_utc: str,
    ) -> dict[str, float] | None:
        """Return the detailed observed stats for a completed fixture.

        The fixture provider includes a kickoff timestamp while the historical
        results feed stores a date, so a small tolerance handles occasional
        provider date corrections without matching a different meeting.
        """
        target = pd.Timestamp(kickoff_utc).tz_localize(None).normalize()
        candidates = self.current[
            (self.current["HomeTeam"] == home_team)
            & (self.current["AwayTeam"] == away_team)
        ].copy()
        if candidates.empty:
            return None
        candidates["date_distance"] = (candidates["Date"].dt.normalize() - target).abs()
        candidates = candidates[candidates["date_distance"] <= pd.Timedelta(days=3)]
        if candidates.empty:
            return None
        row = candidates.sort_values("date_distance").iloc[0]
        output: dict[str, float] = {}
        for label, (home_column, away_column) in STAT_KEYS.items():
            home_value, away_value = row.get(home_column), row.get(away_column)
            if not pd.isna(home_value):
                output[f"home_{label}"] = float(home_value)
            if not pd.isna(away_value):
                output[f"away_{label}"] = float(away_value)
        return output

    def match_center(self, home_team: str, away_team: str) -> dict[str, Any]:
        return {
            "home": self.team_profile(home_team),
            "away": self.team_profile(away_team),
            "head_to_head": self.head_to_head(home_team, away_team),
            "stat_forecast": self.stat_forecast(home_team, away_team),
            "players_to_watch": {
                "available": False,
                "players": [],
                "reason": "The connected data plan does not provide dependable player statistics.",
            },
            "generated_for": datetime.now(UTC).date().isoformat(),
        }
