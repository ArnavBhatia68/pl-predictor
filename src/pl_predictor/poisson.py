from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class IndependentPoissonModel:
    """Predict home/away goal rates and convert them into scoreline probabilities."""

    def __init__(self, model_type: str, params: dict[str, Any] | None = None) -> None:
        self.model_type = model_type
        self.params = params or {}
        self.home_model = self._build_regressor()
        self.away_model = self._build_regressor()

    def _build_regressor(self) -> Pipeline:
        if self.model_type == "linear":
            regressor = PoissonRegressor(
                alpha=float(self.params.get("alpha", 0.3)),
                max_iter=1500,
            )
            preprocess: Any = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                ]
            )
        elif self.model_type == "histogram":
            regressor = HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=float(self.params.get("learning_rate", 0.05)),
                max_iter=int(self.params.get("max_iter", 250)),
                max_leaf_nodes=int(self.params.get("max_leaf_nodes", 15)),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 30)),
                l2_regularization=float(self.params.get("l2_regularization", 5.0)),
                random_state=42,
            )
            preprocess = SimpleImputer(strategy="median", keep_empty_features=True)
        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise RuntimeError('Install V3 dependencies with: pip install -e ".[ml]"') from exc
            regressor = XGBRegressor(
                objective="count:poisson",
                eval_metric="poisson-nloglik",
                max_depth=int(self.params.get("max_depth", 2)),
                learning_rate=float(self.params.get("learning_rate", 0.03)),
                n_estimators=int(self.params.get("n_estimators", 350)),
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_lambda=8.0,
                reg_alpha=0.2,
                max_delta_step=0.7,
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
            preprocess = SimpleImputer(strategy="median", keep_empty_features=True)
        else:
            raise ValueError(f"Unknown Poisson regressor type: {self.model_type}")
        return Pipeline([("preprocess", preprocess), ("regressor", regressor)])

    def fit(
        self,
        features: pd.DataFrame,
        home_goals: pd.Series,
        away_goals: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "IndependentPoissonModel":
        fit_params = {"regressor__sample_weight": sample_weight} if sample_weight is not None else {}
        self.home_model.fit(features, home_goals, **fit_params)
        self.away_model.fit(features, away_goals, **fit_params)
        return self

    def predict_goal_rates(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home = np.clip(self.home_model.predict(features), 0.05, 7.0)
        away = np.clip(self.away_model.predict(features), 0.05, 7.0)
        return home, away

    @staticmethod
    def score_matrices_from_rates(
        home_rates: np.ndarray,
        away_rates: np.ndarray,
        rho: float = 0.0,
        max_goals: int = 10,
    ) -> np.ndarray:
        scores = np.arange(max_goals + 1)
        home_pmfs = poisson.pmf(scores[None, :], home_rates[:, None])
        away_pmfs = poisson.pmf(scores[None, :], away_rates[:, None])
        matrices = home_pmfs[:, :, None] * away_pmfs[:, None, :]

        # Dixon-Coles low-score correction. rho=0 recovers independent Poisson.
        matrices[:, 0, 0] *= np.clip(1.0 - rho * home_rates * away_rates, 1e-8, None)
        matrices[:, 0, 1] *= np.clip(1.0 + rho * home_rates, 1e-8, None)
        matrices[:, 1, 0] *= np.clip(1.0 + rho * away_rates, 1e-8, None)
        matrices[:, 1, 1] *= max(1.0 - rho, 1e-8)
        matrices /= matrices.sum(axis=(1, 2), keepdims=True)
        return matrices

    @staticmethod
    def outcome_probabilities_from_matrices(matrices: np.ndarray) -> np.ndarray:
        away = np.triu(matrices, k=1).sum(axis=(1, 2))
        draw = np.diagonal(matrices, axis1=1, axis2=2).sum(axis=1)
        home = np.tril(matrices, k=-1).sum(axis=(1, 2))
        probabilities = np.column_stack([away, draw, home])
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict_score_matrices(
        self, features: pd.DataFrame, rho: float = 0.0, max_goals: int = 10
    ) -> np.ndarray:
        home_rates, away_rates = self.predict_goal_rates(features)
        return self.score_matrices_from_rates(home_rates, away_rates, rho, max_goals)

    def predict_proba(
        self, features: pd.DataFrame, rho: float = 0.0, max_goals: int = 10
    ) -> np.ndarray:
        matrices = self.predict_score_matrices(features, rho, max_goals)
        return self.outcome_probabilities_from_matrices(matrices)

    def predict_scorelines(
        self, features: pd.DataFrame, rho: float = 0.0, max_goals: int = 10
    ) -> list[str]:
        matrices = self.predict_score_matrices(features, rho, max_goals)
        flat_indices = matrices.reshape(len(matrices), -1).argmax(axis=1)
        home_scores, away_scores = np.divmod(flat_indices, max_goals + 1)
        return [f"{home}-{away}" for home, away in zip(home_scores, away_scores)]


class LogProbabilityCalibrator:
    """Multinomial calibration that works for any model producing class probabilities."""

    def __init__(self) -> None:
        self.model = LogisticRegression(C=10.0, max_iter=2000, random_state=42)

    @staticmethod
    def _features(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-8, 1.0)
        return np.log(clipped)

    def fit(self, probabilities: np.ndarray, targets: pd.Series) -> "LogProbabilityCalibrator":
        self.model.fit(self._features(probabilities), targets)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict_proba(self._features(probabilities))
        indices = [list(self.model.classes_).index(index) for index in (0, 1, 2)]
        return calibrated[:, indices]


class CalibratedPoissonModel:
    def __init__(
        self,
        base_model: IndependentPoissonModel,
        rho: float,
        calibrator: LogProbabilityCalibrator | None,
    ) -> None:
        self.base_model = base_model
        self.rho = rho
        self.calibrator = calibrator

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.base_model.predict_proba(features, rho=self.rho)
        return self.calibrator.predict_proba(raw) if self.calibrator is not None else raw

    def predict_goal_rates(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.base_model.predict_goal_rates(features)

    def predict_scorelines(self, features: pd.DataFrame) -> list[str]:
        return self.base_model.predict_scorelines(features, rho=self.rho)
