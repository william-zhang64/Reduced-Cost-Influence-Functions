"""Clean-ERM oracle, Huber, trimmed ERM and SPO+ baselines (Huber and SPO+: linear model only)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import HuberRegressor

from .lp import LPSolver
from .models import LinearModel


def _require_linear(model):
    if not isinstance(model, LinearModel):
        raise TypeError("this baseline is implemented for the linear upstream model only")


def fit_clean_erm(model, solver, Z_train, y_train, Z_val, y_val, y_train_clean=None, **extra):
    """ERM on clean training labels + validation."""
    if y_train_clean is None:
        raise ValueError("fit_clean_erm needs y_train_clean")
    return model.fit(np.vstack([Z_train, Z_val]), np.vstack([y_train_clean, y_val]),
                     warm_start=False)


def fit_huber(model, solver, Z_train, y_train, Z_val, y_val, epsilon=1.35, **extra):
    """Per-output Huber regression on train + validation."""
    _require_linear(model)
    Z, y = np.vstack([Z_train, Z_val]), np.vstack([y_train, y_val])
    model.fit(Z[:2], y[:2], warm_start=False)
    W = np.zeros((y.shape[1], Z.shape[1]))
    for k in range(y.shape[1]):
        h = HuberRegressor(epsilon=epsilon, fit_intercept=False, alpha=1e-8, max_iter=1000)
        h.fit(Z, y[:, k])
        W[k] = h.coef_
    model.set_theta(W)
    return model


def fit_trimmed(model, solver, Z_train, y_train, Z_val, y_val, trim=0.3, n_rounds=3, **extra):
    """Iteratively trimmed least squares on train + validation."""
    Z, y = np.vstack([Z_train, Z_val]), np.vstack([y_train, y_val])
    model.fit(Z, y, warm_start=False)
    for _ in range(n_rounds):
        R = np.linalg.norm(y - model.predict(Z), axis=1)
        keep = R <= np.quantile(R, 1 - trim)
        model.fit(Z[keep], y[keep], warm_start=False)
    return model


def _spo_plus_loss(c_hat, c, x_true, x_spo):
    return -((2.0 * c_hat - c) @ x_spo) + 2.0 * c_hat @ x_true - c @ x_true


def spo_plus_descent(solver: LPSolver, Z, C, W0, n_epochs=100, lr0=0.5, normalize_grad=True):
    """Full-batch SPO+ subgradient descent for c_hat = W z; returns the best iterate."""
    solve = solver.solve_fast
    n = Z.shape[0]
    W = np.array(W0, dtype=float, copy=True)
    x_true = [solve(C[i]) for i in range(n)]

    def epoch(W, want_grad=True):
        G = np.zeros_like(W) if want_grad else None
        loss, used = 0.0, 0
        for i in range(n):
            if x_true[i] is None:
                continue
            c_hat = W @ Z[i]
            x_spo = solve(2.0 * c_hat - C[i])
            if x_spo is None:
                continue
            loss += _spo_plus_loss(c_hat, C[i], x_true[i], x_spo)
            if want_grad:
                G += np.outer(2.0 * (x_true[i] - x_spo), Z[i])
            used += 1
        if used == 0:
            return None, np.inf
        if want_grad:
            G /= used
        return G, loss / used

    best_W, best_loss = W.copy(), np.inf
    for t in range(n_epochs):
        G, loss = epoch(W)
        if G is None:
            break
        if loss < best_loss:
            best_loss, best_W = loss, W.copy()
        if normalize_grad:
            gn = np.linalg.norm(G)
            if gn > 1e-12:
                G = G / gn
        W = W - lr0 / np.sqrt(t + 1.0) * G
    _, final_loss = epoch(W, want_grad=False)
    if final_loss < best_loss:
        best_W = W.copy()
    return best_W


def fit_spo_plus_val_only(model, solver, Z_train, y_train, Z_val, y_val,
                          n_epochs=100, lr0=0.5, **extra):
    """SPO+ descent on the validation set, initialised at validation-only OLS."""
    _require_linear(model)
    model.fit(Z_val, y_val, warm_start=False)
    model.set_theta(spo_plus_descent(solver, Z_val, y_val, model.theta, n_epochs, lr0))
    return model


def fit_spo_plus_finetune(model, solver, Z_train, y_train, Z_val, y_val,
                          n_epochs=30, lr0=0.1, **extra):
    """OLS on train + validation, then SPO+ descent on the validation set."""
    _require_linear(model)
    model.fit(np.vstack([Z_train, Z_val]), np.vstack([y_train, y_val]), warm_start=False)
    model.set_theta(spo_plus_descent(solver, Z_val, y_val, model.theta, n_epochs, lr0))
    return model
