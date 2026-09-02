from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np

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
    ) -> PredictionService:
        artifact = _load_model_artifact(model_path)
        state = LiveFeatureState.from_csv(matches_path)
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        return cls(artifact, state, metrics)

    def resolve_team(self, name: str) -> str:
        normalized = " ".join(name.lower().strip().split())
        alias = TEAM_ALIASES.get(normalized)
        if alias in self.state.available_teams:
            return alias
        exact = {team.lower(): team for team in self.state.available_teams}
        if normalized in exact:
            return exact[normalized]
        raise ValueError(
            f"Unknown team '{name}'. Use one of: {', '.join(self.state.available_teams)}"
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
        latest = self.state.last_match_date.date() if self.state.last_match_date is not None else date.today()
        fixture_date = fixture_date or max(date.today(), latest + timedelta(days=1))
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
