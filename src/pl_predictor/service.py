from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import MATCHES_PATH, V4_METRICS_PATH, V4_MODEL_PATH, V4_PREDICTIONS_PATH
from .live import LiveFeatureState

TEAM_ALIASES = {
    "arsenal fc": "Arsenal",
    "aston villa fc": "Aston Villa",
    "afc bournemouth": "Bournemouth",
    "brentford fc": "Brentford",
    "brighton & hove albion fc": "Brighton",
    "chelsea fc": "Chelsea",
    "coventry city fc": "Coventry",
    "crystal palace fc": "Crystal Palace",
    "everton fc": "Everton",
    "fulham fc": "Fulham",
    "hull city afc": "Hull",
    "ipswich town fc": "Ipswich",
    "leeds united fc": "Leeds",
    "liverpool fc": "Liverpool",
    "manchester city fc": "Man City",
    "manchester city": "Man City",
    "manchester united fc": "Man United",
    "manchester united": "Man United",
    "man utd": "Man United",
    "newcastle united fc": "Newcastle",
    "nottingham forest fc": "Nott'm Forest",
    "nottingham forest": "Nott'm Forest",
    "nottingham": "Nott'm Forest",
    "sunderland afc": "Sunderland",
    "spurs": "Tottenham",
    "tottenham hotspur fc": "Tottenham",
    "tottenham hotspur": "Tottenham",
}


def _load_model_artifact(model_path: Path) -> dict[str, Any]:
    """Load models created where sklearn's loss extension had a top-level alias."""
    from sklearn._loss import _loss as sklearn_loss_extension

    # The training runtime serialized this Cython extension as ``_loss`` while
    # standard scikit-learn wheels expose it as ``sklearn._loss._loss``.
    sys.modules.setdefault("_loss", sklearn_loss_extension)
    return joblib.load(model_path)


def _load_live_feature_state(
    matches_path: Path,
    state_path: Path,
) -> LiveFeatureState:
    matches = pd.read_csv(matches_path)
    matches["Date"] = pd.to_datetime(matches["Date"], errors="raise")
    latest_match_date = matches["Date"].max()
    match_count = len(matches)

    if state_path.exists():
        state = joblib.load(state_path)
        processed_count = getattr(state, "match_count", None)
        if (
            isinstance(state, LiveFeatureState)
            and state.last_match_date == latest_match_date
            and processed_count == match_count
        ):
            return state
        if (
            isinstance(state, LiveFeatureState)
            and isinstance(processed_count, int)
            and 0 < processed_count < match_count
        ):
            new_matches = matches.iloc[processed_count:].copy()
            if (
                state.last_match_date is not None
                and new_matches["Date"].min() >= state.last_match_date
            ):
                state.replay(new_matches)
                state.match_count = match_count
                try:
                    joblib.dump(state, state_path, compress=3)
                except OSError:
                    pass
                return state

    state = LiveFeatureState().replay(matches)
    state.match_count = match_count
    try:
        joblib.dump(state, state_path, compress=3)
    except OSError:
        # A read-only deployment filesystem should not prevent predictions.
        pass
    return state


class PredictionService:
    def __init__(
        self,
        artifact: dict[str, Any],
        state: LiveFeatureState,
        metrics: dict[str, Any] | None = None,
        prediction_history_path: Path = V4_PREDICTIONS_PATH,
    ) -> None:
        self.artifact = artifact
        self.model = artifact["model"]
        self.state = state
        self.metrics = metrics or {}
        self.prediction_history_path = prediction_history_path

    @classmethod
    def from_paths(
        cls,
        model_path: Path = V4_MODEL_PATH,
        matches_path: Path = MATCHES_PATH,
        metrics_path: Path = V4_METRICS_PATH,
        state_path: Path | None = None,
    ) -> PredictionService:
        artifact = _load_model_artifact(model_path)
        state = _load_live_feature_state(
            matches_path,
            state_path or model_path.with_name("live_feature_state.joblib"),
        )
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        return cls(artifact, state, metrics)

    def resolve_team(self, name: str) -> str:
        return self._canonical_team(name, require_active=True)

    def _canonical_team(self, name: str, *, require_active: bool) -> str:
        normalized = " ".join(name.lower().strip().split())
        alias = TEAM_ALIASES.get(normalized)
        known_teams = set(self.state.available_teams) | set(self.state.elo.ratings)
        if alias and (not require_active or alias in self.state.available_teams):
            return alias
        exact = {team.lower(): team for team in known_teams}
        if normalized in exact:
            candidate = exact[normalized]
            if not require_active or candidate in self.state.available_teams:
                return candidate
        simplified = normalized
        for suffix in (" football club", " fc", " afc"):
            if simplified.endswith(suffix):
                simplified = simplified.removesuffix(suffix).strip()
                break
        if simplified in exact:
            candidate = exact[simplified]
            if not require_active or candidate in self.state.available_teams:
                return candidate
        if not require_active:
            return " ".join(part.capitalize() for part in simplified.split())
        raise ValueError(
            f"Unknown team '{name}'. Use one of: {', '.join(self.state.available_teams)}"
        )

    def prepare_season(self, season_start: int, team_names: list[str]) -> None:
        teams = [
            self._canonical_team(name, require_active=False)
            for name in team_names
        ]
        self.state.prepare_season(season_start, teams)

    def refresh_state(
        self,
        matches_path: Path = MATCHES_PATH,
        state_path: Path | None = None,
    ) -> None:
        self.state = _load_live_feature_state(
            matches_path,
            state_path or V4_MODEL_PATH.with_name("live_feature_state.joblib"),
        )

    @property
    def teams(self) -> list[str]:
        return self.state.available_teams

    @staticmethod
    def _confidence(probabilities: np.ndarray) -> tuple[float, str]:
        maximum = float(probabilities.max())
        if maximum >= 0.60:
            label = "high"
        elif maximum >= 0.45:
            label = "medium"
        else:
            label = "low"
        return maximum, label

    def predict(
        self,
        home_team: str,
        away_team: str,
        fixture_date: date | None = None,
        season_start: int | None = None,
    ) -> dict[str, Any]:
        home = self.resolve_team(home_team)
        away = self.resolve_team(away_team)
        if home == away:
            raise ValueError("Home and away teams must be different")
        today = datetime.now(UTC).date()
        latest = self.state.last_match_date.date() if self.state.last_match_date is not None else today
        fixture_date = fixture_date or max(today, latest + timedelta(days=1))
        frame = self.state.fixture_features(home, away, fixture_date, season_start)

        ensemble = self.model.predict_proba(frame)[0]
        classifier, poisson = self.model.component_probabilities(frame)
        home_rates, away_rates = self.model.predict_goal_rates(frame)
        scoreline = self.model.predict_scorelines(frame)[0]
        label_index = int(ensemble.argmax())
        outcome_labels = [away, "Draw", home]
        confidence, confidence_label = self._confidence(ensemble)

        return {
            "fixture": {
                "home_team": home,
                "away_team": away,
                "date": fixture_date.isoformat(),
                "season": self.state.active_season_label,
            },
            "prediction": {
                "most_likely_outcome": outcome_labels[label_index],
                "most_likely_score": scoreline,
                "confidence": confidence,
                "confidence_label": confidence_label,
            },
            "probabilities": {
                "home_win": float(ensemble[2]),
                "draw": float(ensemble[1]),
                "away_win": float(ensemble[0]),
            },
            "expected_goals": {
                "home": float(home_rates[0]),
                "away": float(away_rates[0]),
            },
            "components": {
                "classifier_weight": float(self.artifact["classifier_weight"]),
                "poisson_weight": float(self.artifact["poisson_weight"]),
                "classifier": {
                    "home_win": float(classifier[0, 2]),
                    "draw": float(classifier[0, 1]),
                    "away_win": float(classifier[0, 0]),
                },
                "poisson": {
                    "home_win": float(poisson[0, 2]),
                    "draw": float(poisson[0, 1]),
                    "away_win": float(poisson[0, 0]),
                },
            },
            "form": {
                "home": {
                    "elo": float(frame.loc[0, "home_elo"]),
                    "points_per_game_last5": float(frame.loc[0, "home_last5_points"]),
                    "goals_for_last5": float(frame.loc[0, "home_last5_goals_for"]),
                    "goals_against_last5": float(frame.loc[0, "home_last5_goals_against"]),
                },
                "away": {
                    "elo": float(frame.loc[0, "away_elo"]),
                    "points_per_game_last5": float(frame.loc[0, "away_last5_points"]),
                    "goals_for_last5": float(frame.loc[0, "away_last5_goals_for"]),
                    "goals_against_last5": float(frame.loc[0, "away_last5_goals_against"]),
                },
            },
            "data_as_of": latest.isoformat(),
            "model_version": "v4-ensemble",
        }

    def performance(self) -> dict[str, Any]:
        test = self.metrics.get("test_ensemble", {})
        return {
            "model_version": "v4-ensemble",
            "classifier_weight": self.artifact.get("classifier_weight"),
            "poisson_weight": self.artifact.get("poisson_weight"),
            "test_season": self.metrics.get("test_season"),
            "accuracy": test.get("accuracy"),
            "log_loss": test.get("log_loss"),
            "brier_score": test.get("brier_score"),
            "macro_f1": test.get("macro_f1"),
        }

    def historical_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.prediction_history_path.exists():
            return []
        import pandas as pd

        frame = pd.read_csv(self.prediction_history_path).tail(limit)
        return frame.replace({np.nan: None}).to_dict(orient="records")
