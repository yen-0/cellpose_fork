import numpy as np


def _bce_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LinearRefiner:
    def __init__(self, n_features=5):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

    def predict_proba(self, x):
        z = x @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y, lr=1e-2, epochs=200):
        losses = []
        best_loss = float("inf")
        best_w = self.w.copy()
        best_b = float(self.b)
        for _ in range(epochs):
            p = self.predict_proba(x)
            loss = _bce_loss(p, y)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_w = self.w.copy()
                best_b = float(self.b)

            grad = p - y
            self.w -= lr * (x.T @ grad) / len(x)
            self.b -= lr * float(np.mean(grad))

        self.w = best_w
        self.b = best_b
        return losses

    def save(self, path):
        np.savez(path, model_type="linear", w=self.w, b=self.b)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        m = cls(n_features=len(data["w"]))
        m.w = data["w"]
        m.b = float(data["b"])
        return m


class NonLinearRefiner:
    def __init__(self, n_features=5, hidden_dim=16):
        self.w1 = (0.05 * np.random.randn(n_features, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = (0.05 * np.random.randn(hidden_dim)).astype(np.float32)
        self.b2 = 0.0

    def _forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        p = 1.0 / (1.0 + np.exp(-z))
        return h, p

    def predict_proba(self, x):
        _, p = self._forward(x)
        return p

    def fit(self, x, y, lr=1e-2, epochs=200):
        losses = []
        best_loss = float("inf")
        best_state = (self.w1.copy(), self.b1.copy(), self.w2.copy(), float(self.b2))
        for _ in range(epochs):
            h, p = self._forward(x)
            loss = _bce_loss(p, y)
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_state = (self.w1.copy(), self.b1.copy(), self.w2.copy(), float(self.b2))

            dz = (p - y) / len(x)
            grad_w2 = h.T @ dz
            grad_b2 = float(np.sum(dz))
            dh = np.outer(dz, self.w2)
            da = dh * (1.0 - h * h)
            grad_w1 = x.T @ da
            grad_b1 = np.sum(da, axis=0)

            self.w2 -= lr * grad_w2.astype(np.float32)
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1.astype(np.float32)
            self.b1 -= lr * grad_b1.astype(np.float32)

        self.w1, self.b1, self.w2, self.b2 = best_state
        return losses

    def save(self, path):
        np.savez(path, model_type="nonlinear", w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)


class RefinerModel:
    @staticmethod
    def load(path):
        data = np.load(path)
        model_type = str(data["model_type"]) if "model_type" in data else "linear"
        if model_type == "nonlinear" and "w1" in data:
            model = NonLinearRefiner(n_features=data["w1"].shape[0], hidden_dim=data["w1"].shape[1])
            model.w1 = data["w1"]
            model.b1 = data["b1"]
            model.w2 = data["w2"]
            model.b2 = float(data["b2"])
            return model
        return LinearRefiner.load(path)
