"""Fit final production artifacts from previously selected configurations."""

from __future__ import annotations

import json

import joblib
import pandas as pd

from .config import (
    MATCHES_PATH,
    MODEL_VERSION,
    V4_METRICS_PATH,
    V4_MODEL_PATH,
    V11_STAT_METRICS_PATH,
    V11_STAT_MODEL_PATH,
    ensure_directories,
)
from .ensemble import ProbabilityEnsemble
from .live import LiveFeatureState
from .stat_models import (
    STAT_TARGETS,
    DetailedStatModels,
    _fit,
    default_stat_candidates,
    load_stat_frame,
)
from .v2 import select_feature_columns
from .v3 import fit_poisson_candidate, goal_feature_columns, load_goal_frame
from .v4 import _fit_nested_classifier, _load_selected_components


def build_production_artifacts(production_season: int | None = None) -> dict[str, object]:
    """Train only the final models; model-selection reports remain immutable."""
    ensure_directories()
    goal_frame = load_goal_frame()
    production_season = production_season or int(goal_frame["season_start"].max())
    classifier_candidate, poisson_candidate, rho = _load_selected_components()
    classifier_features = select_feature_columns(goal_frame, classifier_candidate.feature_set)
    poisson_features = goal_feature_columns(goal_frame)
    v4_metrics = json.loads(V4_METRICS_PATH.read_text(encoding="utf-8"))
    classifier_weight = float(v4_metrics["classifier_weight"])

    classifier = _fit_nested_classifier(
        goal_frame, production_season, classifier_candidate, classifier_features
    )
    poisson_train = goal_frame[goal_frame["season_start"] < production_season]
    poisson = fit_poisson_candidate(poisson_candidate, poisson_train, poisson_features)
    ensemble = ProbabilityEnsemble(
        classifier,
        poisson,
        classifier_features,
        poisson_features,
        classifier_weight,
        rho,
    )
    joblib.dump(
        {
            "model": ensemble,
            "classifier_candidate": classifier_candidate,
            "poisson_candidate": poisson_candidate,
            "classifier_weight": classifier_weight,
            "poisson_weight": 1.0 - classifier_weight,
            "rho": rho,
            "production_season": production_season,
            "model_version": MODEL_VERSION,
        },
        V4_MODEL_PATH,
    )

    stat_frame, stat_features = load_stat_frame()
    stat_metrics = json.loads(V11_STAT_METRICS_PATH.read_text(encoding="utf-8"))
    candidates = {candidate.name: candidate for candidate in default_stat_candidates()}
    production_train = stat_frame[stat_frame["season_start"] < production_season]
    stat_models = {
        metric: _fit(
            candidates[stat_metrics["selected_candidates"][metric]],
            production_train,
            stat_features,
            home_target,
            away_target,
        )
        for metric, (home_target, away_target) in STAT_TARGETS.items()
    }
    joblib.dump(
        DetailedStatModels(
            stat_models,
            stat_features,
            stat_metrics["selected_candidates"],
            stat_metrics,
        ),
        V11_STAT_MODEL_PATH,
        compress=3,
    )

    matches = pd.read_csv(MATCHES_PATH)
    matches["Date"] = pd.to_datetime(matches["Date"], errors="raise")
    state = LiveFeatureState().replay(matches)
    state.match_count = len(matches)
    joblib.dump(state, V4_MODEL_PATH.with_name("live_feature_state.joblib"), compress=3)
    result = {
        "model_version": MODEL_VERSION,
        "production_season": production_season,
        "training_rows": len(production_train),
        "classifier_weight": classifier_weight,
        "poisson_weight": 1.0 - classifier_weight,
        "stat_models": sorted(stat_models),
    }
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    build_production_artifacts()
