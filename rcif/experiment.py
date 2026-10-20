"""Run one configuration over seeds and summarise the results."""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from . import data as D
from .configs import ExperimentConfig
from .lp import LPSolver
from .methods import additional_methods, extra_methods, paper_methods
from .models import LinearModel, TwoLayerNNRegressor


def default_methods(cfg: ExperimentConfig, extra: bool = False, additional: bool = False) -> dict:
    m = paper_methods()
    if extra:
        m.update(extra_methods())
    if additional:
        m.update(additional_methods())
    return m


def build_problem(cfg: ExperimentConfig):
    """(solver, lp_dim, theta_or_None)."""
    if cfg.problem == "random_lp":
        A, b, theta = D.make_random_lp(cfg.num_constraints, cfg.lp_dim, cfg.input_dim,
                                       instance_seed=cfg.instance_seed)
        return LPSolver(A, b), cfg.lp_dim, theta
    if cfg.problem == "shortest_path":
        A, b, n_edges = D.make_shortest_path_lp(cfg.grid)
        return LPSolver(A, b), n_edges, None
    raise ValueError(f"unknown problem {cfg.problem!r}")


def make_split(cfg: ExperimentConfig, seed: int, solver: LPSolver, theta):
    """Train/val/test split for one seed with corrupted training labels."""
    rng = np.random.default_rng(seed)
    n_total = cfg.n_total
    if cfg.problem == "random_lp":
        Z, Y = D.generate_linear_data(n_total, theta, cfg.label_noise, rng)
    else:
        Z, Y = D.generate_shortest_path_data(n_total, cfg.input_dim, grid=cfg.grid,
                                             deg=cfg.deg, noise_width=cfg.noise_width,
                                             seed=seed)
    tr, va = cfg.train_size, cfg.train_size + cfg.val_size
    y_train_clean = Y[:tr].copy()
    n_corrupt = int(cfg.corruption_rate * cfg.train_size)
    idx = rng.choice(cfg.train_size, n_corrupt, replace=False)
    if cfg.corruption == "gaussian":
        y_train = D.corrupt_gaussian(y_train_clean, idx, cfg.noise_scale, rng)
    elif cfg.corruption == "cone_flip":
        y_train = D.corrupt_cone_flip(y_train_clean, idx, solver, cfg.budget)
    elif cfg.corruption == "directional":
        y_train = D.corrupt_directional(y_train_clean, idx, solver, cfg.alpha)
    else:
        raise ValueError(f"unknown corruption {cfg.corruption!r}")
    return dict(
        Z_train=Z[:tr].copy(), y_train=y_train, y_train_clean=y_train_clean,
        Z_val=Z[tr:va].copy(), y_val=Y[tr:va].copy(),
        Z_test=Z[va:].copy(), y_test=Y[va:].copy(), corrupt_idx=idx,
    )


def make_model(cfg: ExperimentConfig, lp_dim: int):
    if cfg.model == "linear":
        return LinearModel()
    if cfg.model == "nn":
        return TwoLayerNNRegressor(cfg.input_dim, lp_dim, hidden_size=cfg.hidden_size,
                                   lr=cfg.nn_lr, max_iter=cfg.nn_max_iter, random_state=0)
    raise ValueError(f"unknown model {cfg.model!r}")


def run_seed(cfg: ExperimentConfig, seed: int, methods: dict | None = None,
             solver=None, lp_dim=None, theta=None, verbose=True) -> dict:
    """{method: test regret} for one seed."""
    if solver is None:
        solver, lp_dim, theta = build_problem(cfg)
    methods = methods or default_methods(cfg)
    split = make_split(cfg, seed, solver, theta)
    args = (split["Z_train"], split["y_train"], split["Z_val"], split["y_val"])
    out = {}
    for name, fit in methods.items():
        t0 = time.time()
        try:
            model = fit(make_model(cfg, lp_dim), solver, *args,
                        y_train_clean=split["y_train_clean"], trim=cfg.corruption_rate)
            out[name] = solver.mean_regret(model.predict(split["Z_test"]), split["y_test"])
        except Exception:
            traceback.print_exc()
            out[name] = np.nan
        if verbose:
            print(f"  seed {seed:3d}  {name:<24s} regret {out[name]:.4f}  ({time.time() - t0:.1f}s)",
                  flush=True)
    return out


def run_experiment(cfg: ExperimentConfig, seeds=None, methods=None, verbose=True) -> pd.DataFrame:
    """DataFrame with columns method, seed, regret."""
    assert cfg.sweep_param is None, "expand() the config before running"
    solver, lp_dim, theta = build_problem(cfg)
    seeds = range(cfg.n_seeds) if seeds is None else seeds
    rows = []
    for seed in seeds:
        res = run_seed(cfg, seed, methods, solver, lp_dim, theta, verbose=verbose)
        rows += [dict(method=m, seed=seed, regret=r) for m, r in res.items()]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean test regret per method with 1.96 * SE."""
    g = df.groupby("method")["regret"].agg(["mean", "std", "count"])
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"])
    g["display"] = [f"{m:.4f} ± {c:.4f}" for m, c in zip(g["mean"], g["ci95"])]
    return g.sort_values("mean")


def paired_wilcoxon(df: pd.DataFrame, baseline: str = "Unweighted") -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test of every method against baseline over seeds."""
    from scipy.stats import wilcoxon

    base = df[df.method == baseline].set_index("seed")["regret"]
    rows = []
    for m in df.method.unique():
        if m == baseline:
            continue
        other = df[df.method == m].set_index("seed")["regret"].reindex(base.index)
        diff = (other - base).dropna()
        p = 1.0 if np.allclose(diff, 0) else wilcoxon(diff).pvalue
        rows.append(dict(method=m, mean_improvement=-diff.mean(),
                         win_rate=(diff < 0).mean(), p_value=p))
    return pd.DataFrame(rows).sort_values("mean_improvement", ascending=False)
