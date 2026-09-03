from __future__ import annotations

import numpy as np
import pandas as pd

from .poisson import IndependentPoissonModel


def blend_probabilities(
    classifier_probabilities: np.ndarray,
    poisson_probabilities: np.ndarray,
    classifier_weight: float,
) -> np.ndarray:
    if not 0.0 <= classifier_weight <= 1.0:
        raise ValueError("classifier_weight must be between 0 and 1")
    if classifier_probabilities.shape != poisson_probabilities.shape:
        raise ValueError("Component probability matrices must have the same shape")
    blended = (
        classifier_weight * classifier_probabilities
        + (1.0 - classifier_weight) * poisson_probabilities
    )
    return blended / blended.sum(axis=1, keepdims=True)


class ProbabilityEnsemble:
    def __init__(
        self,
        classifier_model: object,
        poisson_model: IndependentPoissonModel,
        classifier_features: list[str],
        poisson_features: list[str],
        classifier_weight: float,
        rho: float,
    ) -> None:
        self.classifier_model = classifier_model
        self.poisson_model = poisson_model
        self.classifier_features = classifier_features
        self.poisson_features = poisson_features
        self.classifier_weight = classifier_weight
        self.rho = rho

    def component_probabilities(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        classifier = self.classifier_model.predict_proba(frame[self.classifier_features])
        poisson = self.poisson_model.predict_proba(frame[self.poisson_features], rho=self.rho)
        return classifier, poisson

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        classifier, poisson = self.component_probabilities(frame)
        return blend_probabilities(classifier, poisson, self.classifier_weight)

    def predict_goal_rates(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.poisson_model.predict_goal_rates(frame[self.poisson_features])

    def predict_scorelines(self, frame: pd.DataFrame) -> list[str]:
        return self.poisson_model.predict_scorelines(frame[self.poisson_features], rho=self.rho)

    def scoreline_distributions(
        self, frame: pd.DataFrame, limit: int = 5
    ) -> list[list[dict[str, float | str]]]:
        matrices = self.poisson_model.predict_score_matrices(
            frame[self.poisson_features], rho=self.rho
        )
        output: list[list[dict[str, float | str]]] = []
        width = matrices.shape[2]
        for matrix in matrices:
            indices = np.argsort(matrix, axis=None)[::-1][:limit]
            rows: list[dict[str, float | str]] = []
            for index in indices:
                home, away = divmod(int(index), width)
                rows.append(
                    {
                        "score": f"{home}-{away}",
                        "probability": float(matrix[home, away]),
                    }
                )
            output.append(rows)
        return output
