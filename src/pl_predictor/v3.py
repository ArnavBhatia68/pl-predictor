from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from .config import (
    FEATURES_PATH,
    MATCHES_PATH,
    V3_IMPORTANCE_PATH,
    V3_METRICS_PATH,
    V3_MODEL_PATH,
    V3_PREDICTIONS_PATH,
    V3_WALK_FORWARD_PATH,
    ensure_directories,
)
from .poisson import CalibratedPoissonModel, IndependentPoissonModel, LogProbabilityCalibrator
from .v2 import TARGET_TO_INT, evaluate_probabilities, sample_weights, walk_forward_splits

GOAL_METRICS = {
    "points",
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "goal_diff",
    "shot_accuracy",
    "conversion_rate",
    "shot_dominance",
    "sot_dominance",
}
SCOPES = ("last5", "last10", "venue_last5", "season")
RHO_VALUES = (-0.10, -0.05, 0.0, 0.05, 0.10)


@dataclass(frozen=True)
class PoissonCandidate:
    name: str
    model_type: str
    half_life_years: float | None = None
    params: dict[str, Any] = field(default_factory=dict)


def default_poisson_candidates() -> list[PoissonCandidate]:
    return [
        PoissonCandidate("poisson_linear", "linear", params={"alpha": 0.3}),
        PoissonCandidate(
            "poisson_linear_decay5", "linear", half_life_years=5.0, params={"alpha": 0.3}
        ),
        PoissonCandidate(
            "poisson_xgb_depth2",
            "xgboost",
            params={"max_depth": 2, "learning_rate": 0.03, "n_estimators": 350},
        ),
    ]


def load_goal_frame(
    features_path: Path = FEATURES_PATH, matches_path: Path = MATCHES_PATH
) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    matches = pd.read_csv(matches_path, usecols=[
        "season_start", "season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"
    ])
    keys = ["season_start", "season", "Date", "HomeTeam", "AwayTeam"]
    frame = features.merge(matches, on=keys, how="left", validate="one_to_one")
    if frame[["FTHG", "FTAG"]].isna().any().any():
        raise ValueError("Could not attach goal targets to every feature row")
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["target_int"] = frame["target"].map(TARGET_TO_INT).astype(int)
    frame["target_home_goals"] = frame["FTHG"].astype(int)
    frame["target_away_goals"] = frame["FTAG"].astype(int)
    return frame.sort_values("Date", kind="stable").reset_index(drop=True)


def goal_feature_columns(frame: pd.DataFrame) -> list[str]:
    selected = [column for column in ("home_elo", "away_elo", "diff_elo") if column in frame]
    for scope in SCOPES:
        for metric in GOAL_METRICS:
            for prefix in ("home", "away", "diff"):
                column = f"{prefix}_{scope}_{metric}"
                if column in frame:
                    selected.append(column)
        for prefix in ("home", "away"):
            available = f"{prefix}_{scope}_matches_available"
            if available in frame:
                selected.append(available)
    forbidden = {"target", "target_int", "target_home_goals", "target_away_goals", "FTHG", "FTAG"}
    if forbidden.intersection(selected):
        raise ValueError("Target leakage detected in Poisson feature selection")
    return list(dict.fromkeys(selected))


def fit_poisson_candidate(
    candidate: PoissonCandidate,
    train: pd.DataFrame,
    feature_columns: list[str],
) -> IndependentPoissonModel:
    weights = sample_weights(
        train["Date"], train["target_int"], candidate.half_life_years, draw_weight=1.0
    )
    model = IndependentPoissonModel(candidate.model_type, candidate.params)
    model.fit(
        train[feature_columns],
        train["target_home_goals"],
        train["target_away_goals"],
        sample_weight=weights,
    )
    return model


def goal_metrics(
    frame: pd.DataFrame, home_rates: np.ndarray, away_rates: np.ndarray
) -> dict[str, float]:
    home_actual = frame["target_home_goals"].to_numpy()
    away_actual = frame["target_away_goals"].to_numpy()
    return {
        "home_goal_mae": float(mean_absolute_error(home_actual, home_rates)),
        "away_goal_mae": float(mean_absolute_error(away_actual, away_rates)),
        "mean_goal_mae": float(
            (mean_absolute_error(home_actual, home_rates) + mean_absolute_error(away_actual, away_rates))
            / 2.0
        ),
        "home_poisson_deviance": float(mean_poisson_deviance(home_actual, home_rates)),
        "away_poisson_deviance": float(mean_poisson_deviance(away_actual, away_rates)),
    }


def run_poisson_walk_forward(
    frame: pd.DataFrame,
    candidates: list[PoissonCandidate],
    validation_seasons: tuple[int, ...],
    feature_columns: list[str],
) -> tuple[PoissonCandidate, float, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for season, train, validation in walk_forward_splits(frame, validation_seasons):
            model = fit_poisson_candidate(candidate, train, feature_columns)
            home_rates, away_rates = model.predict_goal_rates(validation[feature_columns])
            rates_metrics = goal_metrics(validation, home_rates, away_rates)
            for rho in RHO_VALUES:
                probabilities = IndependentPoissonModel.outcome_probabilities_from_matrices(
                    IndependentPoissonModel.score_matrices_from_rates(home_rates, away_rates, rho=rho)
                )
                outcome = evaluate_probabilities(validation["target_int"], probabilities)
                row = {
                    "candidate": candidate.name,
                    "model_type": candidate.model_type,
                    "features": len(feature_columns),
                    "validation_season": season,
                    "rho": rho,
                    **{key: outcome[key] for key in (
                        "accuracy", "macro_f1", "log_loss", "brier_score", "ece",
                        "draw_recall", "draw_prediction_rate"
                    )},
                    **rates_metrics,
                }
                rows.append(row)
            best_fold = min(
                (row for row in rows if row["candidate"] == candidate.name and row["validation_season"] == season),
                key=lambda row: row["log_loss"],
            )
            print(
                f"{candidate.name:30s} {season}/{str(season + 1)[-2:]} "
                f"best_rho={best_fold['rho']:+.2f} log_loss={best_fold['log_loss']:.3f} "
                f"accuracy={best_fold['accuracy']:.3f} draw_recall={best_fold['draw_recall']:.3f}"
            )

    folds = pd.DataFrame(rows)
    leaderboard_frame = (
        folds.groupby(["candidate", "model_type", "features", "rho"], as_index=False)
        .agg(
            mean_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_brier=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
            mean_draw_recall=("draw_recall", "mean"),
            mean_draw_prediction_rate=("draw_prediction_rate", "mean"),
            mean_goal_mae=("mean_goal_mae", "mean"),
        )
        .sort_values(["mean_log_loss", "mean_brier"], kind="stable")
        .reset_index(drop=True)
    )
    leaderboard_frame.insert(0, "rank", np.arange(1, len(leaderboard_frame) + 1))
    winner_row = leaderboard_frame.iloc[0]
    winner = next(candidate for candidate in candidates if candidate.name == winner_row["candidate"])
    return winner, float(winner_row["rho"]), rows, leaderboard_frame.to_dict(orient="records")


def _regressor_shap_values(
    pipeline: Any,
    model_type: str,
    sample: pd.DataFrame,
) -> np.ndarray:
    import shap

    transformed = pipeline.named_steps["preprocess"].transform(sample)
    regressor = pipeline.named_steps["regressor"]
    if model_type in {"histogram", "xgboost"}:
        explainer = shap.TreeExplainer(regressor)
    else:
        background = transformed[: min(100, len(transformed))]
        explainer = shap.LinearExplainer(regressor, background)
    values = np.asarray(explainer(transformed).values)
    if values.ndim != 2:
        raise ValueError(f"Unexpected Poisson SHAP shape: {values.shape}")
    return np.abs(values).mean(axis=0)


def global_poisson_shap(
    model: IndependentPoissonModel,
    model_type: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> list[dict[str, Any]]:
    sample = frame[feature_columns].iloc[: min(500, len(frame))]
    try:
        home = _regressor_shap_values(model.home_model, model_type, sample)
        away = _regressor_shap_values(model.away_model, model_type, sample)
    except ImportError:
        def native_importance(pipeline: Any) -> np.ndarray:
            estimator = pipeline.named_steps["regressor"]
            values = getattr(estimator, "feature_importances_", None)
            if values is None:
                values = np.abs(np.asarray(estimator.coef_))
            return np.asarray(values)

        home = native_importance(model.home_model)
        away = native_importance(model.away_model)
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap_home_goals": home,
            "mean_abs_shap_away_goals": away,
            "mean_abs_shap": (home + away) / 2.0,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance.to_csv(output_path, index=False)
    return importance.head(20).to_dict(orient="records")


def run_v3(
    features_path: Path = FEATURES_PATH,
    matches_path: Path = MATCHES_PATH,
    validation_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023),
    calibration_season: int = 2024,
    test_season: int = 2025,
    candidates: list[PoissonCandidate] | None = None,
    model_path: Path = V3_MODEL_PATH,
    metrics_path: Path = V3_METRICS_PATH,
    predictions_path: Path = V3_PREDICTIONS_PATH,
    walk_forward_path: Path = V3_WALK_FORWARD_PATH,
    importance_path: Path = V3_IMPORTANCE_PATH,
) -> dict[str, Any]:
    ensure_directories()
    frame = load_goal_frame(features_path, matches_path)
    features = goal_feature_columns(frame)
    candidates = candidates or default_poisson_candidates()
    winner, rho, fold_rows, leaderboard = run_poisson_walk_forward(
        frame, candidates, validation_seasons, features
    )
    pd.DataFrame(fold_rows).to_csv(walk_forward_path, index=False)
    print(f"Poisson walk-forward winner: {winner.name}, rho={rho:+.2f}")

    # Decide whether calibration generalizes using seasons that precede the final test.
    calibration_choice_base = frame[frame["season_start"] < calibration_season - 1].copy()
    calibration_choice_fit = frame[frame["season_start"] == calibration_season - 1].copy()
    calibration_choice_validation = frame[frame["season_start"] == calibration_season].copy()
    choice_model = fit_poisson_candidate(winner, calibration_choice_base, features)
    choice_fit_raw = choice_model.predict_proba(calibration_choice_fit[features], rho=rho)
    choice_calibrator = LogProbabilityCalibrator().fit(
        choice_fit_raw, calibration_choice_fit["target_int"]
    )
    choice_validation_raw = choice_model.predict_proba(
        calibration_choice_validation[features], rho=rho
    )
    choice_validation_calibrated = choice_calibrator.predict_proba(choice_validation_raw)
    choice_raw_metrics = evaluate_probabilities(
        calibration_choice_validation["target_int"], choice_validation_raw
    )
    choice_calibrated_metrics = evaluate_probabilities(
        calibration_choice_validation["target_int"], choice_validation_calibrated
    )
    use_calibration = (
        choice_calibrated_metrics["log_loss"] < choice_raw_metrics["log_loss"]
    )
    print(
        f"Calibration selection on {calibration_season}/{str(calibration_season + 1)[-2:]}: "
        f"raw={choice_raw_metrics['log_loss']:.3f}, "
        f"calibrated={choice_calibrated_metrics['log_loss']:.3f}; "
        f"use_calibration={use_calibration}"
    )

    base_train = frame[frame["season_start"] < calibration_season].copy()
    calibration = frame[frame["season_start"] == calibration_season].copy()
    test = frame[frame["season_start"] == test_season].copy()
    if base_train.empty or calibration.empty or test.empty:
        raise ValueError("V3 chronological split contains an empty partition")

    base_model = fit_poisson_candidate(winner, base_train, features)
    calibration_raw = base_model.predict_proba(calibration[features], rho=rho)
    calibrator = LogProbabilityCalibrator().fit(calibration_raw, calibration["target_int"])
    production_model = CalibratedPoissonModel(
        base_model, rho, calibrator if use_calibration else None
    )

    raw_probabilities = base_model.predict_proba(test[features], rho=rho)
    calibrated_probabilities = calibrator.predict_proba(raw_probabilities)
    production_probabilities = production_model.predict_proba(test[features])
    raw_metrics = evaluate_probabilities(test["target_int"], raw_probabilities)
    calibrated_metrics = evaluate_probabilities(test["target_int"], calibrated_probabilities)
    home_rates, away_rates = base_model.predict_goal_rates(test[features])
    final_goal_metrics = goal_metrics(test, home_rates, away_rates)

    predictions = test[["season", "Date", "HomeTeam", "AwayTeam", "target"]].copy()
    predictions["expected_home_goals"] = home_rates
    predictions["expected_away_goals"] = away_rates
    predictions["most_likely_score"] = base_model.predict_scorelines(test[features], rho=rho)
    predictions[["prob_away", "prob_draw", "prob_home"]] = production_probabilities
    predictions["prediction"] = np.asarray(["A", "D", "H"])[production_probabilities.argmax(axis=1)]
    predictions["correct"] = predictions["prediction"] == predictions["target"]
    predictions.to_csv(predictions_path, index=False)

    top_features = global_poisson_shap(
        base_model, winner.model_type, test, features, importance_path
    )
    artifact = {
        "model": production_model,
        "base_model": base_model,
        "feature_columns": features,
        "labels": ["A", "D", "H"],
        "selected_candidate": asdict(winner),
        "rho": rho,
        "uses_calibration": use_calibration,
        "calibration_season": calibration_season,
        "test_season": test_season,
    }
    joblib.dump(artifact, model_path)

    metrics = {
        "selected_candidate": asdict(winner),
        "rho": rho,
        "uses_calibration": use_calibration,
        "feature_count": len(features),
        "validation_seasons": list(validation_seasons),
        "calibration_season": calibration_season,
        "test_season": test_season,
        "leaderboard": leaderboard,
        "calibration_selection": {
            "base_train_before": calibration_season - 1,
            "calibration_fit_season": calibration_season - 1,
            "validation_season": calibration_season,
            "raw": choice_raw_metrics,
            "calibrated": choice_calibrated_metrics,
            "selected": "calibrated" if use_calibration else "raw",
        },
        "test_raw": raw_metrics,
        "test_calibrated": calibrated_metrics,
        "test_production": evaluate_probabilities(
            test["target_int"], production_probabilities
        ),
        "test_goal_metrics": final_goal_metrics,
        "top_shap_features": top_features,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Final production Poisson test: accuracy={metrics['test_production']['accuracy']:.3f}, "
        f"log_loss={metrics['test_production']['log_loss']:.3f}, "
        f"draw_recall={metrics['test_production']['draw_recall']:.3f}, "
        f"goal_MAE={final_goal_metrics['mean_goal_mae']:.3f}"
    )
    print(f"Saved V3 model to {model_path}")
    return metrics
