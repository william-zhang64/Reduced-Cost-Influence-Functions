"""python -m pytest tests  (needs a Gurobi licence)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rcif import LPSolver, LinearModel, TwoLayerNNRegressor, rcif_scores  # noqa: E402
from rcif.configs import ExperimentConfig  # noqa: E402
from rcif.data import corrupt_cone_flip, make_random_lp  # noqa: E402
from rcif.experiment import run_seed  # noqa: E402


@pytest.fixture(scope="module")
def problem():
    A, b, theta = make_random_lp(20, 10, 8, instance_seed=1)
    return LPSolver(A, b), theta


def test_reduced_costs_match_edge_directions(problem):
    solver, _ = problem
    rng = np.random.default_rng(0)
    for _ in range(5):
        c = rng.standard_normal(solver.dim)
        sol = solver.solve_with_basis(c)
        for k in np.flatnonzero(sol.nonbasic):
            assert sol.reduced_costs[k] >= -1e-9
            assert abs(c @ sol.directions[k] - sol.reduced_costs[k]) < 1e-6
        assert sol.same_cone(c)
        assert np.allclose(solver.solve(c), sol.x)


def test_cone_flip_changes_decision(problem):
    solver, _ = problem
    rng = np.random.default_rng(1)
    y = rng.standard_normal((6, solver.dim))
    y_c = corrupt_cone_flip(y, np.arange(6), solver, budget=1e6)
    for j in range(6):
        sol = solver.solve_with_basis(y[j])
        assert not sol.same_cone(y_c[j])


def test_linear_influence_matches_finite_difference(problem):
    solver, theta_true = problem
    rng = np.random.default_rng(2)
    n, q, d = 60, theta_true.shape[1], theta_true.shape[0]
    Z = rng.standard_normal((n, q))
    y = Z @ theta_true.T + 0.5 * rng.standard_normal((n, d))
    z_new = rng.standard_normal(q)
    model = LinearModel().fit(Z, y)
    sol = solver.solve_with_basis(model.predict(z_new[None])[0])
    k = sol.fragile_direction()
    delta = sol.directions[k]
    infl = model.influence(Z, y, damp=0.0)
    rcif = infl.rcif_all(delta, z_new)

    eps, j = 1e-4, 7
    w = np.ones(n) / n
    w[j] += eps
    theta_eps = LinearModel().fit(Z, y, sample_weight=w).theta
    fd = (delta @ (theta_eps @ z_new) - delta @ model.predict(z_new[None])[0]) / eps
    assert abs(fd - rcif[j]) < 1e-3 * max(1.0, abs(fd))


def test_rcif_sign_convention(problem):
    solver, theta_true = problem
    rng = np.random.default_rng(3)
    n, q, d = 80, theta_true.shape[1], theta_true.shape[0]
    Z = rng.standard_normal((n, q))
    y = Z @ theta_true.T + 0.3 * rng.standard_normal((n, d))
    model = LinearModel().fit(Z, y)
    s = rcif_scores(model, solver, Z, y, Z[:10], y[:10], scheme="directed")
    assert s.shape == (n,) and np.all(np.isfinite(s))
    s2 = rcif_scores(model, solver, Z, y, Z[:10], y[:10], scheme="two_sided")
    assert s2.shape == (n,)


def test_nn_influence_shape():
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((30, 5))
    y = rng.standard_normal((30, 3))
    model = TwoLayerNNRegressor(5, 3, hidden_size=4, max_iter=20).fit(Z, y, warm_start=False)
    infl = model.influence(Z, y)
    assert infl.shift_all(Z[0]).shape == (30, 3)
    assert infl.rcif_all(np.ones(3), Z[0]).shape == (30,)


def test_end_to_end_small_config():
    cfg = ExperimentConfig(name="tiny", problem="random_lp", input_dim=6, lp_dim=5,
                           num_constraints=8, train_size=40, val_size=10, test_size=10,
                           corruption_rate=0.3, n_seeds=1)
    res = run_seed(cfg, 0, verbose=False)
    assert set(res) >= {"Iterative Exp. Directed", "Unweighted", "Validation Only",
                        "SPO Influence Weight"}
    assert all(np.isfinite(v) and v >= 0 for v in res.values())
