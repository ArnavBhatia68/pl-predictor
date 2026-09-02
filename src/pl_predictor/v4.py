from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .config import (
    FEATURES_PATH,
    MATCHES_PATH,
    V2_METRICS_PATH,
    V3_METRICS_PATH,
    V4_METRICS_PATH,
    V4_MODEL_PATH,
    V4_PREDICTIONS_PATH,
    V4_WALK_FORWARD_PATH,
    ensure_directories,
)
from .ensemble import ProbabilityEnsemble, blend_probabilities
from .v2 import Candidate, evaluate_probabilities, fit_candidate, select_feature_columns
from .v3 import (
    PoissonCandidate,
    fit_poisson_candidate,
    goal_feature_columns,
    load_goal_frame,
)


WEIGHTS = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))


def _load_selected_components(
    v2_metrics_path: Path = V2_METRICS_PATH,
    v3_metrics_path: Path = V3_METRICS_PATH,
) -> tuple[Candidate, PoissonCandidate, float]:
    v2_metrics = json.loads(v2_metrics_path.read_text(encoding="utf-8"))
    v3_metrics = json.loads(v3_metrics_path.read_text(encoding="utf-8"))
    classifier = Candidate(**v2_metrics["selected_candidate"])
    poisson = PoissonCandidate(**v3_metrics["selected_candidate"])
    return classifier, poisson, float(v3_metrics["rho"])


def _fit_nested_classifier(
    frame: pd.DataFrame,
    prediction_season: int,
    candidate: Candidate,
    feature_columns: list[str],
) -> CalibratedClassifierCV:
    calibration_season = prediction_season - 1
    base_train = frame[frame["season_start"] < calibration_season].copy()
    calibration = frame[frame["season_start"] == calibration_season].copy()
    if base_train.empty or calibration.empty:
        raise ValueError(f"Classifier nested split is empty for prediction season {prediction_season}")
    base_model = fit_candidate(candidate, base_train, feature_columns)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(calibration[feature_columns], calibration["target_int"])
    return calibrated


def _component_probabilities_for_season(
    frame: pd.DataFrame,
    season: int,
    classifier_candidate: Candidate,
    poisson_candidate: PoissonCandidate,
    classifier_features: list[str],
    poisson_features: list[str],
    rho: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    validation = frame[frame["season_start"] == season].copy()
    if validation.empty:
        raise ValueError(f"No validation rows for season {season}")
    classifier = _fit_nested_classifier(
        frame, season, classifier_candidate, classifier_features
    )
    poisson_train = frame[frame["season_start"] < season].copy()
    poisson = fit_poisson_candidate(poisson_candidate, poisson_train, poisson_features)
    classifier_probabilities = classifier.predict_proba(validation[classifier_features])
    poisson_probabilities = poisson.predict_proba(validation[poisson_features], rho=rho)
    return classifier_probabilities, poisson_probabilities, validation


def run_ensemble_walk_forward(
    frame: pd.DataFrame,
    validation_seasons: tuple[int, ...],
    classifier_candidate: Candidate,
    poisson_candidate: PoissonCandidate,
    classifier_features: list[str],
    poisson_features: list[str],
    rho: float,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for season in validation_seasons:
        classifier, poisson, validation = _component_probabilities_for_season(
            frame,
            season,
            classifier_candidate,
            poisson_candidate,
            classifier_features,
            poisson_features,
            rho,
        )
        classifier_metrics = evaluate_probabilities(validation["target_int"], classifier)
        poisson_metrics = evaluate_probabilities(validation["target_int"], poisson)
        print(
            f"{season}/{str(season + 1)[-2:]} components: "
            f"classifier={classifier_metrics['log_loss']:.3f}, "
            f"poisson={poisson_metrics['log_loss']:.3f}"
        )
        for weight in WEIGHTS:
            probabilities = blend_probabilities(classifier, poisson, weight)
            metrics = evaluate_probabilities(validation["target_int"], probabilities)
            rows.append(
                {
                    "validation_season": season,
                    "classifier_weight": weight,
                    "poisson_weight": 1.0 - weight,
                    **{key: metrics[key] for key in (
                        "accuracy", "macro_f1", "log_loss", "brier_score", "ece",
                        "draw_recall", "draw_prediction_rate", "actual_draw_rate",
                        "mean_draw_probability"
                    )},
                }
            )

    folds = pd.DataFrame(rows)
    leaderboard_frame = (
        folds.groupby(["classifier_weight", "poisson_weight"], as_index=False)
        .agg(
            mean_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_brier=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
            mean_draw_recall=("draw_recall", "mean"),
            mean_draw_probability=("mean_draw_probability", "mean"),
        )
        .sort_values(["mean_log_loss", "mean_brier"], kind="stable")
        .reset_index(drop=True)
    )
    leaderboard_frame.insert(0, "rank", np.arange(1, len(leaderboard_frame) + 1))
    best_weight = float(leaderboard_frame.iloc[0]["classifier_weight"])
    return best_weight, rows, leaderboard_frame.to_dict(orient="records")


def run_v4(
    features_path: Path = FEATURES_PATH,
    matches_path: Path = MATCHES_PATH,
    validation_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023),
    test_season: int = 2025,
    model_path: Path = V4_MODEL_PATH,
    metrics_path: Path = V4_METRICS_PATH,
    predictions_path: Path = V4_PREDICTIONS_PATH,
    walk_forward_path: Path = V4_WALK_FORWARD_PATH,
) -> dict[str, Any]:
    ensure_directories()
    frame = load_goal_frame(features_path, matches_path)
    classifier_candidate, poisson_candidate, rho = _load_selected_components()
    classifier_features = select_feature_columns(frame, classifier_candidate.feature_set)
    poisson_features = goal_feature_columns(frame)

    weight, fold_rows, leaderboard = run_ensemble_walk_forward(
        frame,
        validation_seasons,
        classifier_candidate,
        poisson_candidate,
        classifier_features,
        poisson_features,
        rho,
    )
    pd.DataFrame(fold_rows).to_csv(walk_forward_path, index=False)
    print(
        f"Selected ensemble weights: classifier={weight:.2f}, poisson={1.0 - weight:.2f}"
    )

    test = frame[frame["season_start"] == test_season].copy()
    classifier_model = _fit_nested_classifier(
        frame, test_season, classifier_candidate, classifier_features
    )
    # The uncalibrated Poisson component can use every season before the test season.
    poisson_train = frame[frame["season_start"] < test_season].copy()
    poisson_model = fit_poisson_candidate(poisson_candidate, poisson_train, poisson_features)
    production_model = ProbabilityEnsemble(
        classifier_model,
        poisson_model,
        classifier_features,
        poisson_features,
        weight,
        rho,
    )
    classifier_probabilities, poisson_probabilities = production_model.component_probabilities(test)
    ensemble_probabilities = production_model.predict_proba(test)
    classifier_metrics = evaluate_probabilities(test["target_int"], classifier_probabilities)
    poisson_metrics = evaluate_probabilities(test["target_int"], poisson_probabilities)
    ensemble_metrics = evaluate_probabilities(test["target_int"], ensemble_probabilities)

    classifier_decisions = classifier_probabilities.argmax(axis=1)
    poisson_decisions = poisson_probabilities.argmax(axis=1)
    disagreement_rate = float(np.mean(classifier_decisions != poisson_decisions))
    mean_probability_distance = float(
        np.mean(np.abs(classifier_probabilities - poisson_probabilities))
    )

    home_rates, away_rates = production_model.predict_goal_rates(test)
    predictions = test[["season", "Date", "HomeTeam", "AwayTeam", "target"]].copy()
    predictions["expected_home_goals"] = home_rates
    predictions["expected_away_goals"] = away_rates
    predictions["most_likely_score"] = production_model.predict_scorelines(test)
    predictions[["v2_prob_away", "v2_prob_draw", "v2_prob_home"]] = classifier_probabilities
    predictions[["v3_prob_away", "v3_prob_draw", "v3_prob_home"]] = poisson_probabilities
    predictions[["prob_away", "prob_draw", "prob_home"]] = ensemble_probabilities
    predictions["prediction"] = np.asarray(["A", "D", "H"])[ensemble_probabilities.argmax(axis=1)]
    predictions["correct"] = predictions["prediction"] == predictions["target"]
    predictions.to_csv(predictions_path, index=False)

    artifact = {
        "model": production_model,
        "classifier_candidate": classifier_candidate,
        "poisson_candidate": poisson_candidate,
        "classifier_weight": weight,
        "poisson_weight": 1.0 - weight,
        "rho": rho,
        "test_season": test_season,
    }
    joblib.dump(artifact, model_path)
    metrics = {
        "validation_seasons": list(validation_seasons),
        "test_season": test_season,
        "classifier_candidate": classifier_candidate.name,
        "poisson_candidate": poisson_candidate.name,
        "rho": rho,
        "classifier_weight": weight,
        "poisson_weight": 1.0 - weight,
        "leaderboard": leaderboard,
        "test_classifier": classifier_metrics,
        "test_poisson": poisson_metrics,
        "test_ensemble": ensemble_metrics,
        "component_disagreement_rate": disagreement_rate,
        "mean_component_probability_distance": mean_probability_distance,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Final ensemble test: accuracy={ensemble_metrics['accuracy']:.3f}, "
        f"log_loss={ensemble_metrics['log_loss']:.3f}, "
        f"draw_probability={ensemble_metrics['mean_draw_probability']:.3f}"
    )
    print(f"Saved V4 model to {model_path}")
    return metrics

