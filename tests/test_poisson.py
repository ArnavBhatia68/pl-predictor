import unittest

import numpy as np

from pl_predictor.poisson import (
    CalibratedPoissonModel,
    IndependentPoissonModel,
    LogProbabilityCalibrator,
)


class PoissonTests(unittest.TestCase):
    def test_outcome_probabilities_sum_to_one(self) -> None:
        matrices = IndependentPoissonModel.score_matrices_from_rates(
            np.array([1.5, 0.8]), np.array([1.0, 1.7]), rho=-0.05
        )
        probabilities = IndependentPoissonModel.outcome_probabilities_from_matrices(matrices)
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        self.assertTrue(np.all(probabilities >= 0))

    def test_stronger_home_rate_increases_home_probability(self) -> None:
        matrices = IndependentPoissonModel.score_matrices_from_rates(
            np.array([2.5]), np.array([0.6])
        )
        away, draw, home = IndependentPoissonModel.outcome_probabilities_from_matrices(matrices)[0]
        self.assertGreater(home, away)
        self.assertGreater(home, draw)

    def test_equal_rates_produce_symmetric_win_probabilities(self) -> None:
        matrices = IndependentPoissonModel.score_matrices_from_rates(
            np.array([1.3]), np.array([1.3])
        )
        away, _, home = IndependentPoissonModel.outcome_probabilities_from_matrices(matrices)[0]
        self.assertAlmostEqual(away, home)

    def test_log_probability_calibrator_preserves_shape(self) -> None:
        probabilities = np.array(
            [
                [0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.6, 0.2],
                [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6],
            ]
        )
        targets = np.array([0, 0, 1, 1, 2, 2])
        calibrator = LogProbabilityCalibrator().fit(probabilities, targets)
        calibrated = calibrator.predict_proba(probabilities)
        self.assertEqual(calibrated.shape, probabilities.shape)
        np.testing.assert_allclose(calibrated.sum(axis=1), np.ones(len(probabilities)))

    def test_production_wrapper_can_leave_probabilities_uncalibrated(self) -> None:
        class StubModel:
            def predict_proba(self, features, rho=0.0):
                return np.array([[0.2, 0.3, 0.5]])

        model = CalibratedPoissonModel(StubModel(), rho=0.05, calibrator=None)
        np.testing.assert_allclose(model.predict_proba(None), np.array([[0.2, 0.3, 0.5]]))


if __name__ == "__main__":
    unittest.main()
