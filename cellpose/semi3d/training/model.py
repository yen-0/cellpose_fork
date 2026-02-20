import numpy as np


class LinearRefiner:
    def __init__(self, n_features=5):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

    def predict_proba(self, x):
        z = x @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y, lr=1e-2, epochs=200):
        for _ in range(epochs):
            p = self.predict_proba(x)
            grad = p - y
            self.w -= lr * (x.T @ grad) / len(x)
            self.b -= lr * float(np.mean(grad))

    def save(self, path):
        np.savez(path, w=self.w, b=self.b)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        m = cls(n_features=len(data["w"]))
        m.w = data["w"]
        m.b = float(data["b"])
        return m
