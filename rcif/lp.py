"""Downstream LP  min c^T x  s.t.  A x <= b, x >= 0: solver, basis, reduced costs, edge directions, regret."""
from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy import sparse
from scipy.sparse.linalg import spsolve

RC_EPS = 1e-10


@dataclass
class LPSolution:
    x: np.ndarray
    reduced_costs: np.ndarray
    nonbasic: np.ndarray
    directions: np.ndarray

    def margins(self) -> np.ndarray:
        """s_k / ||delta(B, k)|| for nonbasic k with reduced cost > RC_EPS, +inf otherwise."""
        out = np.full(len(self.reduced_costs), np.inf)
        for k in np.flatnonzero(self.nonbasic):
            rc = self.reduced_costs[k]
            nrm = np.linalg.norm(self.directions[k])
            if rc > RC_EPS and nrm > RC_EPS:
                out[k] = rc / nrm
        return out

    def fragile_direction(self) -> int | None:
        """Nonbasic direction with the smallest margin, or None."""
        m = self.margins()
        k = int(np.argmin(m))
        return k if np.isfinite(m[k]) else None

    def entering_direction(self, c_true: np.ndarray) -> int | None:
        """Direction scored at an incorrect validation point, or None."""
        m = self.margins()
        ks = np.flatnonzero(np.isfinite(m) & (self.directions @ c_true < 0))
        return int(ks[-1]) if len(ks) else None

    def same_cone(self, c: np.ndarray) -> bool:
        """True if c lies in the optimality cone of this basis."""
        for k in np.flatnonzero(self.nonbasic):
            if c @ self.directions[k] < 0:
                return False
        return True


class LPSolver:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.n_constraints, self.dim = self.A.shape
        self._A_full = sparse.hstack(
            [sparse.csc_matrix(self.A), sparse.eye(self.n_constraints, format="csc")],
            format="csc",
        )
        self._env = gp.Env(params={"OutputFlag": 0})

    def _optimize(self, c: np.ndarray):
        m = gp.Model(env=self._env)
        m.setParam("OutputFlag", 0)
        m.setParam("Method", 0)  # primal simplex (basis information)
        x = m.addMVar(shape=self.dim, lb=0.0)
        constr = m.addConstr(self.A @ x <= self.b)
        m.setObjective(np.asarray(c, dtype=float) @ x, GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return None
        return m, x, constr

    def solve(self, c: np.ndarray) -> np.ndarray | None:
        """Optimal decision x*(c), or None."""
        res = self._optimize(c)
        if res is None:
            return None
        _, x, _ = res
        return np.array(x.X)

    def solve_fast(self, c: np.ndarray) -> np.ndarray | None:
        """Like solve(), re-using one Gurobi model (only the objective changes)."""
        if not hasattr(self, "_fast"):
            m = gp.Model(env=self._env)
            m.setParam("OutputFlag", 0)
            m.setParam("Method", 0)
            x = m.addMVar(shape=self.dim, lb=0.0)
            m.addConstr(self.A @ x <= self.b)
            self._fast = (m, x)
        m, x = self._fast
        m.setObjective(np.asarray(c, dtype=float) @ x, GRB.MINIMIZE)
        m.optimize()
        return np.array(x.X) if m.Status == GRB.OPTIMAL else None

    def solve_with_basis(self, c: np.ndarray) -> LPSolution | None:
        """Optimal decision with reduced costs, nonbasic mask and edge directions."""
        res = self._optimize(c)
        if res is None:
            return None
        _, x, constr = res
        var_basis = np.array(x.VBasis)
        full_status = np.concatenate([var_basis, np.array(constr.CBasis)])
        basic_idx = np.flatnonzero(full_status == 0)
        B = self._A_full[:, basic_idx]
        nonbasic = var_basis != 0

        directions = np.zeros((self.dim, self.dim))
        for k in np.flatnonzero(nonbasic):
            a_k = self._A_full[:, k].toarray().ravel()
            d_B = spsolve(B, -a_k)
            for pos, col in enumerate(basic_idx):
                if col < self.dim:
                    directions[k, col] = d_B[pos]
            directions[k, k] = 1.0

        return LPSolution(
            x=np.array(x.X),
            reduced_costs=np.array(x.RC),
            nonbasic=nonbasic,
            directions=directions,
        )

    def regret(self, c_pred: np.ndarray, c_true: np.ndarray) -> float:
        """c^T x*(c_hat) - c^T x*(c), clipped at zero."""
        x_true = self.solve(c_true)
        x_pred = self.solve(c_pred)
        return max(0.0, float(c_true @ x_pred - c_true @ x_true))

    def mean_regret(self, C_pred: np.ndarray, C_true: np.ndarray) -> float:
        return float(np.mean([self.regret(cp, ct) for cp, ct in zip(C_pred, C_true)]))
