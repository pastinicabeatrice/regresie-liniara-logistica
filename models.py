import numpy as np


def standardize(X):
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)


class GradientDescentModel:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.losses = []
        for _ in range(self.epochs):
            error = self.output(X) - y
            self.weights -= self.learning_rate * X.T @ error / len(y)
            self.bias -= self.learning_rate * error.mean()
            self.losses.append(self.loss(X, y))
        return self

    def linear(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias


class LinearRegression(GradientDescentModel):
    def output(self, X):
        return self.linear(X)

    def predict(self, X):
        return self.output(X)

    def loss(self, X, y):
        return np.mean((self.output(X) - y) ** 2)


class LogisticRegression(GradientDescentModel):
    def output(self, X):
        return 1 / (1 + np.exp(-self.linear(X)))

    def predict(self, X):
        return (self.output(X) >= 0.5).astype(int)

    def loss(self, X, y):
        p = np.clip(self.output(X), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
