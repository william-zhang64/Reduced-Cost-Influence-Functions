"""RCIF scores, reduced-cost margin scores and SPO+ influence scores."""
from __future__ import annotations

import numpy as np

from .lp import LPSolver

SCHEMES = ("two_sided", "directed")


def rcif_scores(model, solver: LPSolver, Z_train, y_train, Z_val, y_val,
                scheme: str = "directed", damp: float = 1e-5) -> np.ndarray:
    """Psi_j for every training point (``two_sided``: fragile direction at every validation
    point; ``directed``: entering direction at incorrect validation points)."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    Z_train = np.asarray(Z_train, dtype=float)
    Z_val = np.asarray(Z_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    infl = model.influence(Z_train, y_train, damp=damp)
    preds = model.predict(Z_val)
    scores = np.zeros(Z_train.shape[0])
    n_used = 0
    for v in range(Z_val.shape[0]):
        sol = solver.solve_with_basis(preds[v])
        if sol is None:
            continue
        n_used += 1
        correct = sol.same_cone(y_val[v])
        if correct or scheme == "two_sided":
            k = sol.fragile_direction()
        else:
            k = sol.entering_direction(y_val[v])
        if k is None:
            continue
        sign = 1.0 if correct else -1.0
        scores += sign * infl.rcif_all(sol.directions[k], Z_val[v])
    return scores / max(n_used, 1)


def rc_margin_scores(model, solver: LPSolver, Z) -> np.ndarray:
    """Smallest positive reduced cost at each point's own prediction."""
    preds = model.predict(np.asarray(Z, dtype=float))
    out = np.zeros(len(preds))
    for j, c_hat in enumerate(preds):
        sol = solver.solve_with_basis(c_hat)
        if sol is None:
            continue
        pos = sol.reduced_costs[sol.reduced_costs > 0]
        if len(pos):
            out[j] = pos.min()
    return out


def spo_plus_influence_scores(model, solver: LPSolver, Z_train, y_train, Z_val, y_val,
                              damp: float = 1e-5) -> np.ndarray:
    """Sigma_j = (1/m) sum_v g_v^T I(w_j) z_v with g_v = 2 (x*(c_v) - x*(2 c_hat_v - c_v))."""
    Z_train = np.asarray(Z_train, dtype=float)
    Z_val = np.asarray(Z_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float)
    infl = model.influence(Z_train, y_train, damp=damp)
    preds = model.predict(Z_val)
    scores = np.zeros(Z_train.shape[0])
    for v in range(Z_val.shape[0]):
        x_true = solver.solve(y_val[v])
        x_spo = solver.solve(2.0 * preds[v] - y_val[v])
        if x_true is None or x_spo is None:
            continue
        g = 2.0 * (x_true - x_spo)
        scores += infl.shift_all(Z_val[v]) @ g
    return scores / Z_val.shape[0]
