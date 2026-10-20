"""Upstream predictors (linear, two-layer NN) and their influence functions."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression


class LinearInfluence:
    """I(w_j) = - r_j ((Sigma_hat + damp/2 I)^{-1} z_j)^T with r_j = theta z_j - c_j."""

    def __init__(self, theta: np.ndarray, Z: np.ndarray, y: np.ndarray, damp: float):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float)
        n, q = Z.shape
        sigma = Z.T @ Z / n
        self._M = np.linalg.inv(sigma + 0.5 * damp * np.eye(q))
        self._residuals = Z @ theta.T - y
        self._ZM = Z @ self._M

    def kappa(self, z: np.ndarray) -> np.ndarray:
        return self._ZM @ np.asarray(z, dtype=float)

    def shift_all(self, z: np.ndarray) -> np.ndarray:
        """(n, d) array whose row j is I(w_j) z."""
        return -self._residuals * self.kappa(z)[:, None]

    def rcif_all(self, direction: np.ndarray, z: np.ndarray) -> np.ndarray:
        """direction^T I(w_j) z for every training point j."""
        return -(self._residuals @ np.asarray(direction, dtype=float)) * self.kappa(z)


class LinearModel:
    def __init__(self):
        self._reg = LinearRegression(fit_intercept=False)

    @property
    def theta(self) -> np.ndarray:
        return self._reg.coef_

    def set_theta(self, theta: np.ndarray):
        self._reg.coef_ = np.asarray(theta, dtype=float)
        return self

    def fit(self, X, y, sample_weight=None, warm_start=True):
        self._reg.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._reg.predict(np.asarray(X, dtype=float))

    def influence(self, Z_train, y_train, damp: float = 1e-5) -> LinearInfluence:
        return LinearInfluence(self.theta, Z_train, y_train, damp)


class _TwoLayerNet(nn.Module):
    def __init__(self, num_features, num_outputs, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


class NNInfluence:
    """Gauss-Newton influence: H = (2/n) sum_i J_i^T J_i + damp I, I(w_j) = -H^{-1} grad ||f(z_j) - c_j||^2."""

    def __init__(self, model: "TwoLayerNNRegressor", Z: np.ndarray, y: np.ndarray, damp: float):
        self.model = model
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float)
        n = Z.shape[0]
        preds = model.predict(Z)
        p = model.num_params
        grads = np.zeros((n, p))
        H = damp * np.eye(p)
        for i in range(n):
            J_i = model.output_jacobian(Z[i])
            grads[i] = 2.0 * ((preds[i] - y[i]) @ J_i)
            H += 2.0 * (J_i.T @ J_i) / n
        try:
            self.param_influence = -np.linalg.solve(H, grads.T).T
        except np.linalg.LinAlgError:
            self.param_influence = -(np.linalg.pinv(H) @ grads.T).T

    def shift_all(self, z: np.ndarray) -> np.ndarray:
        J = self.model.output_jacobian(z)
        return self.param_influence @ J.T

    def rcif_all(self, direction: np.ndarray, z: np.ndarray) -> np.ndarray:
        J = self.model.output_jacobian(z)
        return self.param_influence @ (np.asarray(direction, dtype=float) @ J)


class TwoLayerNNRegressor:
    """Two-layer tanh MLP trained by full-batch Adam on (weighted) squared loss."""

    def __init__(self, num_features, num_outputs, hidden_size=16, lr=1e-3,
                 max_iter=500, weight_decay=0.0, random_state=0,
                 dtype=torch.float64):
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size
        self.lr = lr
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.dtype = dtype
        self.network = None
        self._jacobian_cache = {}
        self._reset_network()

    def _reset_network(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.network = _TwoLayerNet(self.num_features, self.num_outputs,
                                    self.hidden_size).to(dtype=self.dtype)
        self._jacobian_cache = {}

    def _tensor(self, arr):
        return torch.as_tensor(np.asarray(arr), dtype=self.dtype)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.network.parameters())

    def fit(self, X, y, sample_weight=None, warm_start=True):
        if not warm_start or self.network is None:
            self._reset_network()
        X_t, y_t = self._tensor(X), self._tensor(y)
        if sample_weight is None:
            w_t = torch.ones(X_t.shape[0], dtype=self.dtype)
        else:
            sw = np.asarray(sample_weight, dtype=float)
            if np.sum(sw) <= 0:
                sw = np.ones_like(sw)
            w_t = self._tensor(sw)
        opt = torch.optim.Adam(self.network.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        total = torch.sum(w_t).clamp_min(1e-12)
        for _ in range(self.max_iter):
            opt.zero_grad()
            pred = self.network(X_t)
            loss = torch.sum(w_t * torch.sum((pred - y_t) ** 2, dim=1)) / total
            loss.backward()
            opt.step()
        self._jacobian_cache = {}
        return self

    def predict(self, X):
        X_t = self._tensor(X)
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        with torch.no_grad():
            return self.network(X_t).detach().cpu().numpy()

    def output_jacobian(self, z) -> np.ndarray:
        """(d, p) Jacobian of the network output w.r.t. all parameters at z."""
        z_arr = np.ascontiguousarray(np.asarray(z, dtype=float).reshape(-1))
        key = z_arr.tobytes()
        if key in self._jacobian_cache:
            return self._jacobian_cache[key]
        params = [p for p in self.network.parameters() if p.requires_grad]
        pred = self.network(self._tensor(z_arr.reshape(1, -1))).reshape(-1)
        rows = []
        for o in range(self.num_outputs):
            grads = torch.autograd.grad(pred[o], params, retain_graph=True)
            rows.append(torch.cat([g.reshape(-1) for g in grads]).detach().cpu().numpy())
        J = np.vstack(rows)
        self._jacobian_cache[key] = J
        return J

    def influence(self, Z_train, y_train, damp: float = 1e-5) -> NNInfluence:
        return NNInfluence(self, Z_train, y_train, damp)
