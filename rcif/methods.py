"""Fitting procedures: fit(model, solver, Z_train, y_train, Z_val, y_val, **extra) -> model."""
from __future__ import annotations

import numpy as np

from .scoring import rc_margin_scores, rcif_scores, spo_plus_influence_scores


def _stack(Z_train, y_train, Z_val, y_val):
    return np.vstack([Z_train, Z_val]), np.vstack([y_train, y_val])


def fit_unweighted(model, solver, Z_train, y_train, Z_val, y_val, **extra):
    """ERM on train + validation."""
    Z, y = _stack(Z_train, y_train, Z_val, y_val)
    return model.fit(Z, y, warm_start=False)


def fit_validation_only(model, solver, Z_train, y_train, Z_val, y_val, **extra):
    """ERM on the validation set alone."""
    return model.fit(Z_val, y_val, warm_start=False)


def fit_rc_weight(model, solver, Z_train, y_train, Z_val, y_val, lam=0.05, **extra):
    """Weights exp(lam * min_k s_k) from each point's own reduced-cost margin, train + validation."""
    Z, y = _stack(Z_train, y_train, Z_val, y_val)
    model.fit(Z, y, warm_start=False)
    w = np.exp(lam * rc_margin_scores(model, solver, Z))
    return model.fit(Z, y, sample_weight=w)


def fit_rcif_additive(model, solver, Z_train, y_train, Z_val, y_val,
                      scheme="directed", phi=0.05, **extra):
    """Weights p_j = max(1 + phi * Psi_j, 0), validation weight 1."""
    model.fit(Z_train, y_train, warm_start=False)
    s = rcif_scores(model, solver, Z_train, y_train, Z_val, y_val, scheme=scheme)
    w = np.maximum(1.0 + phi * s, 0.0)
    Z, y = _stack(Z_train, y_train, Z_val, y_val)
    return model.fit(Z, y, sample_weight=np.concatenate([w, np.ones(len(Z_val))]))


def fit_rcif_exp_iterative(model, solver, Z_train, y_train, Z_val, y_val,
                           scheme="directed", lam=0.05, n_iter=5, **extra):
    """Weights exp(lam * sum_t Psi^(t)) normalised to sum to n, validation weight max_j p_j."""
    model.fit(Z_train, y_train, warm_start=False)
    n = len(Z_train)
    Z, y = _stack(Z_train, y_train, Z_val, y_val)
    cumulative = np.zeros(n)
    for _ in range(n_iter):
        cumulative += rcif_scores(model, solver, Z_train, y_train, Z_val, y_val, scheme=scheme)
        w = np.exp(lam * cumulative)
        w = w / w.sum() * n
        model.fit(Z, y, sample_weight=np.concatenate([w, np.full(len(Z_val), w.max())]))
    return model


def fit_rcif_exp(model, solver, Z_train, y_train, Z_val, y_val,
                 scheme="two_sided", lam=0.05, **extra):
    """Single round of fit_rcif_exp_iterative."""
    return fit_rcif_exp_iterative(model, solver, Z_train, y_train, Z_val, y_val,
                                  scheme=scheme, lam=lam, n_iter=1)


def fit_spo_influence(model, solver, Z_train, y_train, Z_val, y_val,
                      lr=0.01, n_iter=5, **extra):
    """Weights p <- max(0, p - lr * Sigma) normalised to sum to n, validation weight 1."""
    model.fit(Z_train, y_train, warm_start=False)
    n = len(Z_train)
    Z, y = _stack(Z_train, y_train, Z_val, y_val)
    w = np.ones(n)
    for _ in range(n_iter):
        d = spo_plus_influence_scores(model, solver, Z_train, y_train, Z_val, y_val)
        w = np.maximum(w - lr * d, 0.0)
        w = w / w.sum() * n if w.sum() > 0 else np.ones(n)
        model.fit(Z, y, sample_weight=np.concatenate([w, np.ones(len(Z_val))]))
    return model


def _bind(fn, **fixed):
    return lambda *args, **kw: fn(*args, **{**fixed, **kw})


def paper_methods() -> dict:
    return {
        "Iterative Exp. Directed": _bind(fit_rcif_exp_iterative, scheme="directed",
                                         lam=0.05, n_iter=5),
        "Exp. Two-Sided": _bind(fit_rcif_additive, scheme="two_sided", phi=0.05),
        "Directed": _bind(fit_rcif_additive, scheme="directed", phi=0.05),
        "Unweighted": fit_unweighted,
        "RC Weight": fit_rc_weight,
        "Validation Only": fit_validation_only,
        "SPO Influence Weight": fit_spo_influence,
    }


def extra_methods() -> dict:
    return {
        "Exp. Two-Sided (exp)": _bind(fit_rcif_exp, scheme="two_sided", lam=0.05),
        "Exp. Directed": _bind(fit_rcif_exp, scheme="directed", lam=0.05),
        "Iterative Exp. Two-Sided": _bind(fit_rcif_exp_iterative, scheme="two_sided",
                                          lam=0.05, n_iter=5),
    }


def additional_methods() -> dict:
    from . import baselines as B

    return {
        "Clean ERM (oracle)": B.fit_clean_erm,
        "Huber": B.fit_huber,
        "Trimmed ERM": B.fit_trimmed,
        "SPO+ Val Only": B.fit_spo_plus_val_only,
        "SPO+ Finetune": B.fit_spo_plus_finetune,
    }
