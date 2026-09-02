from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES_PATH, METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH, ensure_directories
from .features import model_feature_columns


CLASSES = ["A", "D", "H"]


def multiclass_brier(y_true: pd.Series, probabilities: np.ndarray, classes: list[str]) -> float:
    class_to_index = {label: index for index, label in enumerate(classes)}
    observed = np.zeros_like(probabilities, dtype=float)
    for row_index, label in enumerate(y_true):
        observed[row_index, class_to_index[str(label)]] = 1.0
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def _models(feature_columns: list[str]) -> dict[str, Pipeline]:
    numeric_logistic = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), feature_columns)],
        remainder="drop",
    )
    numeric_tree = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), feature_columns)],
        remainder="drop",
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", numeric_logistic),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", numeric_tree),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=8,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _probabilities_in_class_order(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    model_classes = list(model.named_steps["model"].classes_)
    indices = [model_classes.index(label) for label in CLASSES]
    return probabilities[:, indices]


def evaluate(model: Pipeline, frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    y_true = frame["target"]
    probabilities = _probabilities_in_class_order(model, frame[feature_columns])
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=CLASSES)),
        "brier_score": multiclass_brier(y_true, probabilities, CLASSES),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            y_true, predicted, labels=CLASSES, output_dict=True, zero_division=0
        ),
    }


def train_models(
    features_path: Path = FEATURES_PATH,
    validation_season: int = 2023,
    test_seasons: tuple[int, ...] = (2024, 2025),
    exclude_seasons: tuple[int, ...] = (2026,),
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
) -> dict[str, Any]:
    ensure_directories()
    frame = pd.read_csv(features_path)
    frame = frame[~frame["season_start"].isin(exclude_seasons)].copy()
    feature_columns = model_feature_columns(frame)

    train = frame[frame["season_start"] < validation_season]
    validation = frame[frame["season_start"] == validation_season]
    test = frame[frame["season_start"].isin(test_seasons)]
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Time split produced an empty partition. Check validation/test seasons and downloaded data."
        )

    candidates = _models(feature_columns)
    validation_metrics: dict[str, dict[str, Any]] = {}
    for name, model in candidates.items():
        model.fit(train[feature_columns], train["target"])
        validation_metrics[name] = evaluate(model, validation, feature_columns)
        print(
            f"{name}: validation accuracy={validation_metrics[name]['accuracy']:.3f}, "
            f"log_loss={validation_metrics[name]['log_loss']:.3f}"
        )

    winner = min(validation_metrics, key=lambda name: validation_metrics[name]["log_loss"])
    final_model = clone(candidates[winner])
    train_plus_validation = pd.concat([train, validation], ignore_index=True)
    final_model.fit(train_plus_validation[feature_columns], train_plus_validation["target"])
    test_metrics = evaluate(final_model, test, feature_columns)

    artifact = {
        "model": final_model,
        "feature_columns": feature_columns,
        "classes": CLASSES,
        "selected_model": winner,
        "trained_through_season": validation_season,
        "test_seasons": list(test_seasons),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    probabilities = _probabilities_in_class_order(final_model, test[feature_columns])
    predictions = test[["season", "Date", "HomeTeam", "AwayTeam", "target"]].copy()
    predictions[["prob_away", "prob_draw", "prob_home"]] = probabilities
    predictions["prediction"] = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    predictions["correct"] = predictions["prediction"] == predictions["target"]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    metrics = {
        "selected_model": winner,
        "class_order": CLASSES,
        "split": {
            "train_before": validation_season,
            "validation_season": validation_season,
            "test_seasons": list(test_seasons),
            "excluded_seasons": list(exclude_seasons),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Selected {winner}; test accuracy={test_metrics['accuracy']:.3f}, "
        f"log_loss={test_metrics['log_loss']:.3f}"
    )
    print(f"Saved model to {model_path}")
    return metrics

