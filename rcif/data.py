"""Problem instances, data generation and label corruption."""
from __future__ import annotations

import numpy as np

from .lp import RC_EPS, LPSolver


def make_random_lp(num_constraints: int, lp_dim: int, input_dim: int,
                   instance_seed: int = 42, rhs: float = 5.0):
    """A ~ U[0,1), b = rhs * 1, theta* ~ N(0, 1)."""
    rng = np.random.default_rng(instance_seed)
    A = rng.random((num_constraints, lp_dim))
    theta = rng.standard_normal((lp_dim, input_dim))
    return A, np.full(num_constraints, rhs), theta


def generate_linear_data(n: int, theta: np.ndarray, sigma: float, rng: np.random.Generator):
    """z ~ N(0, I_q), c = theta z + sigma * N(0, I_d)."""
    d, q = theta.shape
    Z = rng.standard_normal((n, q))
    C = Z @ theta.T + sigma * rng.standard_normal((n, d))
    return Z, C


def make_shortest_path_lp(grid=(4, 4)):
    """(A, b, num_edges) for PyEPO's shortest-path model in the form A x <= b."""
    from gurobipy import GRB
    from pyepo.model.grb import shortestPathModel

    model = shortestPathModel(grid)._model
    model.update()
    A_raw = model.getA().toarray()
    rows_A, rows_b = [], []
    for i, con in enumerate(model.getConstrs()):
        row, rhs = A_raw[i:i + 1], con.RHS
        if con.Sense == GRB.EQUAL:
            rows_A += [row, -row]
            rows_b += [rhs, -rhs]
        elif con.Sense == GRB.LESS_EQUAL:
            rows_A.append(row)
            rows_b.append(rhs)
        elif con.Sense == GRB.GREATER_EQUAL:
            rows_A.append(-row)
            rows_b.append(-rhs)
        else:
            raise ValueError(f"unsupported constraint sense {con.Sense!r}")
    n_var = model.NumVars
    for j, var in enumerate(model.getVars()):
        if np.isfinite(var.UB):
            row = np.zeros((1, n_var))
            row[0, j] = 1.0
            rows_A.append(row)
            rows_b.append(float(var.UB))
    return np.vstack(rows_A), np.asarray(rows_b, dtype=float), n_var


def generate_shortest_path_data(n: int, input_dim: int, grid=(4, 4), deg: int = 1,
                                noise_width: float = 0.5, seed: int = 0):
    """PyEPO shortest-path features and edge costs."""
    import pyepo

    feats, costs = pyepo.data.shortestpath.genData(
        num_data=n, num_features=input_dim, grid=grid, deg=deg,
        noise_width=noise_width, seed=seed,
    )
    return np.asarray(feats, dtype=float), np.asarray(costs, dtype=float)


def corrupt_gaussian(y: np.ndarray, idx: np.ndarray, noise_scale: float,
                     rng: np.random.Generator) -> np.ndarray:
    """c_j + noise_scale * N(0, I) on rows idx."""
    out = y.copy()
    out[idx] += noise_scale * rng.standard_normal((len(idx), y.shape[1]))
    return out


def corrupt_cone_flip(y: np.ndarray, idx: np.ndarray, solver: LPSolver,
                      budget: float, slack: float = 0.01) -> np.ndarray:
    """Minimum-norm shift across the nearest cone facet (full budget along it if it does not fit)."""
    out = y.copy()
    for j in idx:
        sol = solver.solve_with_basis(y[j])
        if sol is None:
            continue
        k = sol.fragile_direction()
        if k is None:
            continue
        d = sol.directions[k]
        unit = d / np.linalg.norm(d)
        flip_cost = sol.reduced_costs[k] / np.linalg.norm(d)
        alpha = flip_cost * (1.0 + slack)
        if alpha < budget:
            out[j] = y[j] - alpha * unit
        else:
            out[j] = y[j] - budget * unit
    return out


def corrupt_directional(y: np.ndarray, idx: np.ndarray, solver: LPSolver,
                        alpha: float) -> np.ndarray:
    """Shift of fixed magnitude alpha along the most fragile direction."""
    out = y.copy()
    for j in idx:
        sol = solver.solve_with_basis(y[j])
        if sol is None:
            continue
        k = sol.fragile_direction()
        if k is None:
            continue
        d = sol.directions[k]
        out[j] = y[j] - alpha * d / np.linalg.norm(d)
    return out


__all__ = [
    "RC_EPS", "make_random_lp", "generate_linear_data", "make_shortest_path_lp",
    "generate_shortest_path_data", "corrupt_gaussian", "corrupt_cone_flip",
    "corrupt_directional",
]
