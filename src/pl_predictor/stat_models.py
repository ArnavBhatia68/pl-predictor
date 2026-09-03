from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from .config import (
    FEATURES_PATH,
    MATCHES_PATH,
    V11_STAT_METRICS_PATH,
    V11_STAT_MODEL_PATH,
    V11_STAT_PREDICTIONS_PATH,
    V11_STAT_WALK_FORWARD_PATH,
    ensure_directories,
)
from .poisson import IndependentPoissonModel
from .v2 import TARGET_TO_INT, sample_weights, select_feature_columns

STAT_TARGETS: dict[str, tuple[str, str]] = {
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
    "corners": ("HC", "AC"),
    "fouls": ("HF", "AF"),
    "yellow_cards": ("HY", "AY"),
}
STAT_MAXIMUMS = {
    "shots": 45.0,
    "shots_on_target": 25.0,
    "corners": 25.0,
    "fouls": 35.0,
    "yellow_cards": 12.0,
}


@dataclass(frozen=True)
class StatCandidate:
    name: str
    model_type: str
    params: dict[str, Any]


def default_stat_candidates() -> list[StatCandidate]:
    return [
        StatCandidate("poisson_linear", "linear", {"alpha": 0.3}),
        StatCandidate(
            "poisson_histogram",
            "histogram",
            {
                "max_leaf_nodes": 15,
                "learning_rate": 0.05,
                "max_iter": 220,
                "min_samples_leaf": 35,
                "l2_regularization": 5.0,
            },
        ),
    ]


def load_stat_frame(
    features_path: Path = FEATURES_PATH,
    matches_path: Path = MATCHES_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(features_path)
    feature_columns = select_feature_columns(features, "compact")
    target_columns = sorted({column for pair in STAT_TARGETS.values() for column in pair})
    matches = pd.read_csv(
        matches_path,
        usecols=["season_start", "season", "Date", "HomeTeam", "AwayTeam", *target_columns],
    )
    keys = ["season_start", "season", "Date", "HomeTeam", "AwayTeam"]
    frame = features.merge(matches, on=keys, how="left", validate="one_to_one")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["target_int"] = frame["target"].map(TARGET_TO_INT).astype(int)
    if frame[target_columns].isna().any().any():
        raise ValueError("Detailed-stat targets are missing from the training dataset")
    forbidden = set(target_columns) | {"FTHG", "FTAG", "target", "target_int"}
    if forbidden.intersection(feature_columns):
        raise ValueError("Target leakage detected in detailed-stat features")
    return frame.sort_values("Date", kind="stable").reset_index(drop=True), feature_columns


def _fit(
    candidate: StatCandidate,
    train: pd.DataFrame,
    feature_columns: list[str],
    home_target: str,
    away_target: str,
) -> IndependentPoissonModel:
    weights = sample_weights(train["Date"], train["target_int"], None, 1.0)
    return IndependentPoissonModel(candidate.model_type, candidate.params).fit(
        train[feature_columns],
        train[home_target],
        train[away_target],
        sample_weight=weights,
    )


def _metrics(
    actual_home: pd.Series,
    actual_away: pd.Series,
    predicted_home: np.ndarray,
    predicted_away: np.ndarray,
) -> dict[str, float]:
    actual = np.concatenate([actual_home.to_numpy(dtype=float), actual_away.to_numpy(dtype=float)])
    predicted = np.clip(np.concatenate([predicted_home, predicted_away]), 0.01, None)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "poisson_deviance": float(mean_poisson_deviance(actual, predicted)),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(predicted.mean()),
    }


def _predict_rates(
    model: IndependentPoissonModel,
    metric: str,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = STAT_MAXIMUMS[metric]
    home = np.clip(model.home_model.predict(features), 0.01, maximum)
    away = np.clip(model.away_model.predict(features), 0.01, maximum)
    return home, away


class DetailedStatModels:
    def __init__(
        self,
        models: dict[str, IndependentPoissonModel],
        feature_columns: list[str],
        selected_candidates: dict[str, str],
        metrics: dict[str, Any],
    ) -> None:
        self.models = models
        self.feature_columns = feature_columns
        self.selected_candidates = selected_candidates
        self.metrics = metrics

    @staticmethod
    def _interval(rate: float) -> list[float]:
        lower, upper = poisson.ppf([0.10, 0.90], max(rate, 0.01))
        return [float(lower), float(upper)]

    def predict(self, frame: pd.DataFrame) -> dict[str, Any]:
        forecasts: dict[str, Any] = {}
        raw: dict[str, tuple[float, float]] = {}
        for metric, model in self.models.items():
            home_values, away_values = _predict_rates(
                model, metric, frame[self.feature_columns]
            )
            raw[metric] = (float(home_values[0]), float(away_values[0]))

        # A shot on target is necessarily also a shot. Keep the independently
        # trained rate model, then enforce the physical relationship at serving.
        if "shots" in raw and "shots_on_target" in raw:
            shots_home, shots_away = raw["shots"]
            sot_home, sot_away = raw["shots_on_target"]
            raw["shots_on_target"] = (min(sot_home, shots_home), min(sot_away, shots_away))

        for metric, (home, away) in raw.items():
            forecasts[metric] = {
                "home": round(home, 1),
                "away": round(away, 1),
                "total": round(home + away, 1),
                "home_interval_80": self._interval(home),
                "away_interval_80": self._interval(away),
                "model": self.selected_candidates[metric],
                "method": "separate leak-free count model trained on chronological pre-match features",
            }
        forecasts["possession"] = {
            "available": False,
            "reason": "The historical training source does not contain possession labels.",
        }
        return forecasts


def train_stat_models(
    features_path: Path = FEATURES_PATH,
    matches_path: Path = MATCHES_PATH,
    validation_seasons: tuple[int, ...] = (2022, 2023, 2024),
    test_season: int = 2025,
    production_season: int = 2026,
    model_path: Path = V11_STAT_MODEL_PATH,
    metrics_path: Path = V11_STAT_METRICS_PATH,
    predictions_path: Path = V11_STAT_PREDICTIONS_PATH,
    walk_forward_path: Path = V11_STAT_WALK_FORWARD_PATH,
) -> dict[str, Any]:
    ensure_directories()
    frame, feature_columns = load_stat_frame(features_path, matches_path)
    candidates = default_stat_candidates()
    fold_rows: list[dict[str, Any]] = []
    selected: dict[str, StatCandidate] = {}

    for metric, (home_target, away_target) in STAT_TARGETS.items():
        for candidate in candidates:
            for season in validation_seasons:
                train = frame[frame["season_start"] < season]
                validation = frame[frame["season_start"] == season]
                model = _fit(candidate, train, feature_columns, home_target, away_target)
                home, away = _predict_rates(model, metric, validation[feature_columns])
                fold_rows.append(
                    {
                        "metric": metric,
                        "candidate": candidate.name,
                        "validation_season": season,
                        **_metrics(validation[home_target], validation[away_target], home, away),
                    }
                )
        leaderboard = (
            pd.DataFrame(row for row in fold_rows if row["metric"] == metric)
            .groupby("candidate", as_index=False)
            .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), poisson_deviance=("poisson_deviance", "mean"))
            .sort_values(["mae", "poisson_deviance"], kind="stable")
        )
        winner_name = str(leaderboard.iloc[0]["candidate"])
        selected[metric] = next(candidate for candidate in candidates if candidate.name == winner_name)
        print(f"{metric}: selected {winner_name} with validation MAE={leaderboard.iloc[0]['mae']:.3f}")

    pd.DataFrame(fold_rows).to_csv(walk_forward_path, index=False)
    test = frame[frame["season_start"] == test_season].copy()
    test_predictions = test[["season", "Date", "HomeTeam", "AwayTeam"]].copy()
    test_metrics: dict[str, Any] = {}
    production_models: dict[str, IndependentPoissonModel] = {}
    production_train = frame[frame["season_start"] < production_season]

    for metric, (home_target, away_target) in STAT_TARGETS.items():
        candidate = selected[metric]
        evaluation_train = frame[frame["season_start"] < test_season]
        evaluation_model = _fit(
            candidate, evaluation_train, feature_columns, home_target, away_target
        )
        home, away = _predict_rates(evaluation_model, metric, test[feature_columns])
        test_metrics[metric] = _metrics(test[home_target], test[away_target], home, away)
        test_predictions[f"actual_home_{metric}"] = test[home_target].to_numpy()
        test_predictions[f"actual_away_{metric}"] = test[away_target].to_numpy()
        test_predictions[f"predicted_home_{metric}"] = home
        test_predictions[f"predicted_away_{metric}"] = away
        production_models[metric] = _fit(
            candidate, production_train, feature_columns, home_target, away_target
        )

    test_predictions.to_csv(predictions_path, index=False)
    leaderboard_rows = (
        pd.DataFrame(fold_rows)
        .groupby(["metric", "candidate"], as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), poisson_deviance=("poisson_deviance", "mean"))
        .sort_values(["metric", "mae"], kind="stable")
        .to_dict(orient="records")
    )
    metrics = {
        "model_version": "v11",
        "feature_count": len(feature_columns),
        "validation_seasons": list(validation_seasons),
        "test_season": test_season,
        "production_trained_through_season": production_season - 1,
        "selected_candidates": {metric: candidate.name for metric, candidate in selected.items()},
        "leaderboard": leaderboard_rows,
        "test": test_metrics,
    }
    artifact = DetailedStatModels(
        production_models,
        feature_columns,
        metrics["selected_candidates"],
        metrics,
    )
    joblib.dump(artifact, model_path, compress=3)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {len(production_models)} detailed-stat models to {model_path}")
    return metrics
