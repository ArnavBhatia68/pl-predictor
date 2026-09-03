from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURES_PATH,
    V2_IMPORTANCE_PATH,
    V2_METRICS_PATH,
    V2_MODEL_PATH,
    V2_PREDICTIONS_PATH,
    V2_WALK_FORWARD_PATH,
    ensure_directories,
)
from .features import model_feature_columns

LABELS = ["A", "D", "H"]
TARGET_TO_INT = {label: index for index, label in enumerate(LABELS)}
INT_TO_TARGET = {index: label for label, index in TARGET_TO_INT.items()}

COMPACT_METRICS = {
    "points",
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "win",
    "goal_diff",
    "shot_accuracy",
    "conversion_rate",
    "shot_dominance",
    "sot_dominance",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    model_type: str
    feature_set: str
    half_life_years: float | None = None
    draw_weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


def default_candidates() -> list[Candidate]:
    return [
        Candidate("lr_compact", "logistic", "compact", params={"C": 0.4}),
        Candidate(
            "xgb_compact_depth2",
            "xgboost",
            "compact",
            params={"max_depth": 2, "learning_rate": 0.04, "n_estimators": 300},
        ),
        Candidate(
            "xgb_compact_decay5",
            "xgboost",
            "compact",
            half_life_years=5.0,
            params={"max_depth": 2, "learning_rate": 0.03, "n_estimators": 350},
        ),
    ]


def select_feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    available = model_feature_columns(frame)
    if feature_set == "full":
        return available
    if feature_set != "compact":
        raise ValueError(f"Unknown feature set: {feature_set}")

    selected = [column for column in ("home_elo", "away_elo", "diff_elo") if column in frame]
    for column in available:
        if not column.startswith("diff_"):
            continue
        if column.endswith("matches_available"):
            continue
        if any(column.endswith(f"_{metric}") for metric in COMPACT_METRICS):
            selected.append(column)

    # The number of prior matches tells the model when rolling statistics are immature.
    selected.extend(
        column
        for column in available
        if column.endswith("matches_available") and column.startswith(("home_", "away_"))
    )
    return list(dict.fromkeys(selected))


def sample_weights(
    dates: pd.Series,
    targets: pd.Series,
    half_life_years: float | None,
    draw_weight: float,
) -> np.ndarray:
    weights = np.ones(len(dates), dtype=float)
    parsed_dates = pd.to_datetime(dates)
    if half_life_years is not None:
        age_years = (parsed_dates.max() - parsed_dates).dt.days.to_numpy(dtype=float) / 365.25
        weights *= np.power(0.5, age_years / half_life_years)
    if draw_weight != 1.0:
        weights *= np.where(targets.to_numpy(dtype=int) == TARGET_TO_INT["D"], draw_weight, 1.0)
    return weights / weights.mean()


def build_model(candidate: Candidate) -> Pipeline:
    if candidate.model_type == "logistic":
        preprocess = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        model = LogisticRegression(
            C=float(candidate.params.get("C", 1.0)),
            max_iter=2500,
            random_state=42,
        )
    elif candidate.model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError('Install V2 dependencies with: pip install -e ".[ml]"') from exc

        preprocess = SimpleImputer(strategy="median", keep_empty_features=True)
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            max_depth=int(candidate.params.get("max_depth", 3)),
            learning_rate=float(candidate.params.get("learning_rate", 0.03)),
            n_estimators=int(candidate.params.get("n_estimators", 350)),
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=8.0,
            reg_alpha=0.2,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {candidate.model_type}")
    return Pipeline([("preprocess", preprocess), ("model", model)])


def fit_candidate(
    candidate: Candidate,
    train: pd.DataFrame,
    feature_columns: list[str],
) -> Pipeline:
    model = build_model(candidate)
    weights = sample_weights(
        train["Date"], train["target_int"], candidate.half_life_years, candidate.draw_weight
    )
    model.fit(train[feature_columns], train["target_int"], model__sample_weight=weights)
    return model


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    observed = np.eye(len(LABELS), dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for lower, upper in pairwise(edges):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error if total else np.nan)


def evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    truth = y_true.to_numpy(dtype=int)
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1, 2])
    draw_total = int(matrix[1].sum())
    return {
        "rows": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1, 2])),
        "brier_score": multiclass_brier(truth, probabilities),
        "ece": expected_calibration_error(truth, probabilities),
        "draw_recall": float(matrix[1, 1] / draw_total) if draw_total else 0.0,
        "draw_prediction_rate": float(np.mean(predictions == TARGET_TO_INT["D"])),
        "actual_draw_rate": float(np.mean(truth == TARGET_TO_INT["D"])),
        "mean_draw_probability": float(probabilities[:, TARGET_TO_INT["D"]].mean()),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=[0, 1, 2],
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def walk_forward_splits(
    frame: pd.DataFrame, validation_seasons: tuple[int, ...]
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    splits = []
    for season in validation_seasons:
        train = frame[frame["season_start"] < season].copy()
        validation = frame[frame["season_start"] == season].copy()
        if train.empty or validation.empty:
            raise ValueError(f"Walk-forward season {season} produced an empty partition")
        if train["Date"].max() >= validation["Date"].min():
            raise ValueError(f"Temporal overlap detected for validation season {season}")
        splits.append((season, train, validation))
    return splits


def run_walk_forward(
    frame: pd.DataFrame,
    candidates: list[Candidate],
    validation_seasons: tuple[int, ...],
) -> tuple[Candidate, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    splits = walk_forward_splits(frame, validation_seasons)
    for candidate in candidates:
        feature_columns = select_feature_columns(frame, candidate.feature_set)
        for season, train, validation in splits:
            model = fit_candidate(candidate, train, feature_columns)
            probabilities = model.predict_proba(validation[feature_columns])
            metrics = evaluate_probabilities(validation["target_int"], probabilities)
            row = {
                "candidate": candidate.name,
                "model_type": candidate.model_type,
                "feature_set": candidate.feature_set,
                "features": len(feature_columns),
                "validation_season": season,
                **{key: metrics[key] for key in (
                    "accuracy", "macro_f1", "log_loss", "brier_score", "ece",
                    "draw_recall", "draw_prediction_rate"
                )},
            }
            rows.append(row)
            print(
                f"{candidate.name:32s} {season}/{str(season + 1)[-2:]} "
                f"log_loss={metrics['log_loss']:.3f} accuracy={metrics['accuracy']:.3f} "
                f"draw_recall={metrics['draw_recall']:.3f}"
            )

    folds = pd.DataFrame(rows)
    leaderboard_frame = (
        folds.groupby(["candidate", "model_type", "feature_set", "features"], as_index=False)
        .agg(
            mean_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_brier=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
            mean_draw_recall=("draw_recall", "mean"),
            mean_draw_prediction_rate=("draw_prediction_rate", "mean"),
        )
        .sort_values(["mean_log_loss", "mean_brier"], kind="stable")
        .reset_index(drop=True)
    )
    leaderboard_frame.insert(0, "rank", np.arange(1, len(leaderboard_frame) + 1))
    leaderboard = leaderboard_frame.to_dict(orient="records")
    winning_name = str(leaderboard_frame.iloc[0]["candidate"])
    winner = next(candidate for candidate in candidates if candidate.name == winning_name)
    return winner, rows, leaderboard


def _global_shap_importance(
    base_model: Pipeline,
    candidate: Candidate,
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> list[dict[str, Any]]:
    sample = frame[feature_columns].iloc[: min(500, len(frame))]
    transformed = base_model.named_steps["preprocess"].transform(sample)
    estimator = base_model.named_steps["model"]
    try:
        import shap

        if candidate.model_type == "xgboost":
            explainer = shap.TreeExplainer(estimator)
        else:
            background = transformed[: min(200, len(transformed))]
            explainer = shap.LinearExplainer(estimator, background)
        values = np.asarray(explainer(transformed).values)
    except ImportError:
        # Deployment training intentionally keeps SHAP optional. Native model
        # importance still provides a deterministic global explanation.
        native = getattr(estimator, "feature_importances_", None)
        if native is None:
            native = np.abs(np.asarray(estimator.coef_)).mean(axis=0)
        values = np.asarray(native)

    per_class: np.ndarray | None = None
    if values.ndim == 1:
        overall = np.abs(values)
    elif values.ndim == 2:
        overall = np.abs(values).mean(axis=0)
    elif values.ndim == 3 and values.shape[1] == len(feature_columns):
        overall = np.abs(values).mean(axis=(0, 2))
        per_class = np.abs(values).mean(axis=0)
    elif values.ndim == 3 and values.shape[2] == len(feature_columns):
        overall = np.abs(values).mean(axis=(0, 1))
        per_class = np.abs(values).mean(axis=1).T
    else:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    importance = pd.DataFrame({"feature": feature_columns, "mean_abs_shap": overall})
    if per_class is not None and per_class.shape[1] == len(LABELS):
        for index, label in enumerate(LABELS):
            importance[f"mean_abs_shap_{label}"] = per_class[:, index]
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance.head(20).to_dict(orient="records")


def run_v2(
    features_path: Path = FEATURES_PATH,
    validation_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023),
    calibration_season: int = 2024,
    test_season: int = 2025,
    candidates: list[Candidate] | None = None,
    model_path: Path = V2_MODEL_PATH,
    metrics_path: Path = V2_METRICS_PATH,
    predictions_path: Path = V2_PREDICTIONS_PATH,
    walk_forward_path: Path = V2_WALK_FORWARD_PATH,
    importance_path: Path = V2_IMPORTANCE_PATH,
) -> dict[str, Any]:
    ensure_directories()
    frame = pd.read_csv(features_path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["target_int"] = frame["target"].map(TARGET_TO_INT)
    if frame["target_int"].isna().any():
        raise ValueError("Feature data contains an unknown target label")
    frame["target_int"] = frame["target_int"].astype(int)
    frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)

    candidates = candidates or default_candidates()
    winner, fold_rows, leaderboard = run_walk_forward(frame, candidates, validation_seasons)
    pd.DataFrame(fold_rows).to_csv(walk_forward_path, index=False)
    print(f"Walk-forward winner: {winner.name}")

    feature_columns = select_feature_columns(frame, winner.feature_set)
    base_train = frame[frame["season_start"] < calibration_season].copy()
    calibration = frame[frame["season_start"] == calibration_season].copy()
    test = frame[frame["season_start"] == test_season].copy()
    if base_train.empty or calibration.empty or test.empty:
        raise ValueError("Final chronological train/calibration/test split contains an empty partition")

    base_model = fit_candidate(winner, base_train, feature_columns)
    uncalibrated_probabilities = base_model.predict_proba(test[feature_columns])
    uncalibrated_metrics = evaluate_probabilities(test["target_int"], uncalibrated_probabilities)

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated_model.fit(calibration[feature_columns], calibration["target_int"])
    calibrated_probabilities = calibrated_model.predict_proba(test[feature_columns])
    calibrated_metrics = evaluate_probabilities(test["target_int"], calibrated_probabilities)

    predictions = test[["season", "Date", "HomeTeam", "AwayTeam", "target"]].copy()
    predictions[["prob_away", "prob_draw", "prob_home"]] = calibrated_probabilities
    predictions["prediction"] = [INT_TO_TARGET[index] for index in calibrated_probabilities.argmax(axis=1)]
    predictions["correct"] = predictions["prediction"] == predictions["target"]
    predictions.to_csv(predictions_path, index=False)

    top_features = _global_shap_importance(
        base_model, winner, test, feature_columns, importance_path
    )
    artifact = {
        "model": calibrated_model,
        "base_model": base_model,
        "feature_columns": feature_columns,
        "labels": LABELS,
        "target_to_int": TARGET_TO_INT,
        "selected_candidate": asdict(winner),
        "trained_before_season": calibration_season,
        "calibration_season": calibration_season,
        "test_season": test_season,
    }
    joblib.dump(artifact, model_path)

    metrics = {
        "selected_candidate": asdict(winner),
        "feature_count": len(feature_columns),
        "validation_seasons": list(validation_seasons),
        "calibration_season": calibration_season,
        "test_season": test_season,
        "leaderboard": leaderboard,
        "test_uncalibrated": uncalibrated_metrics,
        "test_calibrated": calibrated_metrics,
        "top_shap_features": top_features,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Final calibrated test: accuracy={calibrated_metrics['accuracy']:.3f}, "
        f"log_loss={calibrated_metrics['log_loss']:.3f}, "
        f"draw_recall={calibrated_metrics['draw_recall']:.3f}"
    )
    print(f"Saved V2 model to {model_path}")
    return metrics
