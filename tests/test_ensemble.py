import unittest

import numpy as np

from pl_predictor.ensemble import blend_probabilities


class EnsembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
        self.poisson = np.array([[0.3, 0.3, 0.4], [0.5, 0.3, 0.2]])

    def test_zero_weight_returns_poisson(self) -> None:
        np.testing.assert_allclose(
            blend_probabilities(self.classifier, self.poisson, 0.0), self.poisson
        )

    def test_one_weight_returns_classifier(self) -> None:
        np.testing.assert_allclose(
            blend_probabilities(self.classifier, self.poisson, 1.0), self.classifier
        )

    def test_blended_probabilities_sum_to_one(self) -> None:
        blended = blend_probabilities(self.classifier, self.poisson, 0.35)
        np.testing.assert_allclose(blended.sum(axis=1), np.ones(len(blended)))

    def test_invalid_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            blend_probabilities(self.classifier, self.poisson, 1.1)


if __name__ == "__main__":
    unittest.main()
