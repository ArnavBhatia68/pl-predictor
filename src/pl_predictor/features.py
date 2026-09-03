from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES_PATH, MATCHES_PATH, ensure_directories
from .elo import EloRatings

RAW_STAT_COLUMNS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
BASE_METRICS = [
    "points",
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "corners_for",
    "corners_against",
    "fouls",
    "yellow_cards",
    "red_cards",
    "win",
]


def _safe_float(value: object) -> float:
    if pd.isna(value):
        return np.nan
    return float(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


@dataclass
class TeamState:
    recent: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=10))
    home: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=5))
    away: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=5))
    season: list[dict[str, float]] = field(default_factory=list)
    previous_season: list[dict[str, float]] = field(default_factory=list)
    last_season_seen: int | None = None

    def begin_season(self, season_start: int) -> None:
        """Roll the season window while retaining genuinely recent PL form."""
        if self.last_season_seen == season_start:
            return
        if self.last_season_seen == season_start - 1:
            self.previous_season = list(self.season)
        else:
            # A promoted/returning club must not inherit stale PL form from years ago.
            self.previous_season = []
            self.recent.clear()
            self.home.clear()
            self.away.clear()
        self.season = []
        self.last_season_seen = season_start


def _nan_sum(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or np.isnan(array).all():
        return np.nan
    return float(np.nansum(array))


def _mean(records: list[dict[str, float]], metric: str) -> float:
    if not records:
        return np.nan
    values = np.asarray([record.get(metric, np.nan) for record in records], dtype=float)
    if np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def summarize(records: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(records)
    summary = {metric: _mean(rows, metric) for metric in BASE_METRICS}
    summary["goal_diff"] = summary["goals_for"] - summary["goals_against"]

    shots = _nan_sum(record.get("shots_for", np.nan) for record in rows)
    shots_against = _nan_sum(record.get("shots_against", np.nan) for record in rows)
    sot = _nan_sum(record.get("sot_for", np.nan) for record in rows)
    sot_against = _nan_sum(record.get("sot_against", np.nan) for record in rows)
    goals = _nan_sum(record.get("goals_for", np.nan) for record in rows)
    corners = _nan_sum(record.get("corners_for", np.nan) for record in rows)
    corners_against = _nan_sum(record.get("corners_against", np.nan) for record in rows)

    summary["shot_accuracy"] = _safe_ratio(sot, shots)
    summary["conversion_rate"] = _safe_ratio(goals, shots)
    summary["shot_dominance"] = _safe_ratio(shots, shots + shots_against)
    summary["sot_dominance"] = _safe_ratio(sot, sot + sot_against)
    summary["corner_dominance"] = _safe_ratio(corners, corners + corners_against)
    summary["matches_available"] = float(len(rows))
    return summary


def smoothed_season_summary(
    state: TeamState,
    league_prior: dict[str, float] | None,
    prior_matches: float = 5.0,
) -> dict[str, float]:
    """Blend immature season-to-date numbers with a leak-free prior.

    Returning clubs use their immediately preceding PL season; promoted clubs
    use the previous season's league average. The prior fades naturally as the
    current season supplies more matches.
    """
    current = summarize(state.season)
    prior = summarize(state.previous_season) if state.previous_season else league_prior
    current_matches = float(len(state.season))
    if prior is None:
        return current
    blended: dict[str, float] = {}
    for metric in current:
        if metric == "matches_available":
            blended[metric] = current_matches
            continue
        current_value = current[metric]
        prior_value = prior.get(metric, np.nan)
        if pd.isna(current_value):
            blended[metric] = float(prior_value)
        elif pd.isna(prior_value):
            blended[metric] = float(current_value)
        else:
            blended[metric] = float(
                (current_matches * current_value + prior_matches * prior_value)
                / (current_matches + prior_matches)
            )
    return blended


def _team_match_record(row: pd.Series, is_home: bool) -> dict[str, float]:
    home_goals = int(row["FTHG"])
    away_goals = int(row["FTAG"])
    goals_for = home_goals if is_home else away_goals
    goals_against = away_goals if is_home else home_goals
    points = 3.0 if goals_for > goals_against else 1.0 if goals_for == goals_against else 0.0

    own = "H" if is_home else "A"
    opponent = "A" if is_home else "H"
    return {
        "points": points,
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "shots_for": _safe_float(row[f"{own}S"]),
        "shots_against": _safe_float(row[f"{opponent}S"]),
        "sot_for": _safe_float(row[f"{own}ST"]),
        "sot_against": _safe_float(row[f"{opponent}ST"]),
        "corners_for": _safe_float(row[f"{own}C"]),
        "corners_against": _safe_float(row[f"{opponent}C"]),
        "fouls": _safe_float(row[f"{own}F"]),
        "yellow_cards": _safe_float(row[f"{own}Y"]),
        "red_cards": _safe_float(row[f"{own}R"]),
        "win": 1.0 if goals_for > goals_against else 0.0,
    }


def _add_summary(features: dict[str, object], prefix: str, summary: dict[str, float]) -> None:
    for metric, value in summary.items():
        features[f"{prefix}_{metric}"] = value


def _add_differences(features: dict[str, object]) -> None:
    # Create home-minus-away differences only for paired numeric features.
    home_keys = [key for key in features if key.startswith("home_")]
    for home_key in home_keys:
        away_key = "away_" + home_key.removeprefix("home_")
        if away_key not in features:
            continue
        home_value = features[home_key]
        away_value = features[away_key]
        if isinstance(home_value, (int, float, np.integer, np.floating)) and isinstance(
            away_value, (int, float, np.integer, np.floating)
        ):
            features[f"diff_{home_key.removeprefix('home_')}"] = float(home_value) - float(away_value)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season_start",
        "season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        *RAW_STAT_COLUMNS,
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Match dataframe is missing columns: {missing}")

    ordered = matches.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="raise")
    ordered = ordered.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="stable").reset_index(drop=True)

    elo = EloRatings()
    team_states: defaultdict[str, TeamState] = defaultdict(TeamState)
    output: list[dict[str, object]] = []
    active_season: int | None = None
    previous_league_summary: dict[str, float] | None = None

    for _, row in ordered.iterrows():
        season_start = int(row["season_start"])
        if active_season is None or season_start != active_season:
            if active_season is not None:
                elo.regress_to_mean()
                previous_records = [
                    record
                    for state in team_states.values()
                    for record in state.season
                    if state.last_season_seen == active_season
                ]
                previous_league_summary = summarize(previous_records)
            active_season = season_start

        home_team = str(row["HomeTeam"])
        away_team = str(row["AwayTeam"])
        home_state = team_states[home_team]
        away_state = team_states[away_team]
        home_state.begin_season(season_start)
        away_state.begin_season(season_start)

        features: dict[str, object] = {
            "season_start": season_start,
            "season": row["season"],
            "Date": row["Date"],
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "target": row["FTR"],
            "home_elo": elo.get(home_team),
            "away_elo": elo.get(away_team),
        }

        _add_summary(features, "home_last5", summarize(list(home_state.recent)[-5:]))
        _add_summary(features, "away_last5", summarize(list(away_state.recent)[-5:]))
        _add_summary(features, "home_last10", summarize(home_state.recent))
        _add_summary(features, "away_last10", summarize(away_state.recent))
        _add_summary(features, "home_venue_last5", summarize(home_state.home))
        _add_summary(features, "away_venue_last5", summarize(away_state.away))
        _add_summary(
            features,
            "home_season",
            smoothed_season_summary(home_state, previous_league_summary),
        )
        _add_summary(
            features,
            "away_season",
            smoothed_season_summary(away_state, previous_league_summary),
        )
        _add_differences(features)
        output.append(features)

        # Only after the feature row is complete do we reveal this match to team state.
        home_record = _team_match_record(row, is_home=True)
        away_record = _team_match_record(row, is_home=False)
        home_state.recent.append(home_record)
        away_state.recent.append(away_record)
        home_state.home.append(home_record)
        away_state.away.append(away_record)
        home_state.season.append(home_record)
        away_state.season.append(away_record)
        elo.update(home_team, away_team, int(row["FTHG"]), int(row["FTAG"]))

    return pd.DataFrame(output)


def build_feature_dataset(
    matches_path: Path = MATCHES_PATH,
    output_path: Path = FEATURES_PATH,
) -> pd.DataFrame:
    ensure_directories()
    matches = pd.read_csv(matches_path)
    features = build_features(matches)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    numeric_features = len(model_feature_columns(features))
    print(f"Saved {len(features):,} rows with {numeric_features} model features to {output_path}")
    return features


def model_feature_columns(features: pd.DataFrame) -> list[str]:
    metadata = {
        "season_start",
        "season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "target",
        "target_int",
    }
    return [column for column in features.columns if column not in metadata]
