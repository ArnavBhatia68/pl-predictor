from __future__ import annotations

import math


class EloRatings:
    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 24.0,
        home_advantage: float = 60.0,
        season_regression: float = 0.20,
    ) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.season_regression = season_regression
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def expected_home_score(self, home_team: str, away_team: str) -> float:
        home = self.get(home_team) + self.home_advantage
        away = self.get(away_team)
        return 1.0 / (1.0 + 10.0 ** ((away - home) / 400.0))

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        home_before = self.get(home_team)
        away_before = self.get(away_team)
        expected_home = self.expected_home_score(home_team, away_team)

        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        margin_multiplier = 1.0 + 0.35 * math.log1p(abs(home_goals - away_goals))
        change = self.k_factor * margin_multiplier * (actual_home - expected_home)
        self.ratings[home_team] = home_before + change
        self.ratings[away_team] = away_before - change

    def regress_to_mean(self) -> None:
        for team, rating in self.ratings.items():
            self.ratings[team] = rating + self.season_regression * (self.initial_rating - rating)

