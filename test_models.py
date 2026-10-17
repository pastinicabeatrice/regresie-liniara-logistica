import unittest

import numpy as np

from models import LinearRegression, LogisticRegression, standardize


class LinearRegressionTest(unittest.TestCase):
    def test_learns_exact_line(self):
        X = np.arange(-5, 6).reshape(-1, 1)
        model = LinearRegression(learning_rate=0.05, epochs=2000).fit(X, 3 * X[:, 0] + 2)
        self.assertAlmostEqual(model.weights[0], 3, places=3)
        self.assertAlmostEqual(model.bias, 2, places=3)

    def test_loss_decreases(self):
        X = standardize(np.arange(20).reshape(-1, 1))
        model = LinearRegression(epochs=50).fit(X, np.arange(20))
        self.assertTrue(all(a >= b for a, b in zip(model.losses, model.losses[1:])))


class LogisticRegressionTest(unittest.TestCase):
    def test_separates_two_classes(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(-2, 1, (50, 2)), rng.normal(2, 1, (50, 2))])
        y = np.array([0] * 50 + [1] * 50)
        model = LogisticRegression(epochs=500).fit(X, y)
        self.assertGreater(np.mean(model.predict(X) == y), 0.95)

    def test_outputs_probabilities(self):
        model = LogisticRegression(epochs=100).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        probabilities = model.output([[-10], [1.5], [10]])
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))
        self.assertLess(probabilities[0], 0.5)
        self.assertGreater(probabilities[2], 0.5)


if __name__ == "__main__":
    unittest.main()
