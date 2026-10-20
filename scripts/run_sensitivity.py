#!/usr/bin/env python
"""(lambda, tau) grid for Iterative Exp. Directed; writes results/sensitivity_raw.csv,
tables/sensitivity.tex and figures/sensitivity_lambda_tau.png.

    python scripts/run_sensitivity.py --lambdas 0.01 0.05 0.1 0.2 0.5 --taus 3 5 10 --seed-range 50 100
"""
import argparse
import itertools
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scipy.stats import wilcoxon  # noqa: E402

from rcif import PAPER_CONFIGS  # noqa: E402
from rcif.experiment import build_problem, make_model, make_split  # noqa: E402
from rcif.methods import fit_rcif_exp_iterative, fit_unweighted  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="table1_shortest_path")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.5])
    p.add_argument("--taus", type=int, nargs="+", default=[3, 5, 10])
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--seed-range", type=int, nargs=2, metavar=("START", "STOP"),
                   help="run seeds START..STOP-1")
    p.add_argument("--results", default="results")
    p.add_argument("--tables", default="tables")
    p.add_argument("--figures", default="figures")
    args = p.parse_args()

    cfg = PAPER_CONFIGS[args.config]
    solver, lp_dim, theta = build_problem(cfg)
    for d in (args.results, args.tables, args.figures):
        os.makedirs(d, exist_ok=True)
    rows, t0 = [], time.time()
    seeds = range(*args.seed_range) if args.seed_range else range(args.seeds)
    for seed in seeds:
        sp = make_split(cfg, seed, solver, theta)
        data = (sp["Z_train"], sp["y_train"], sp["Z_val"], sp["y_val"])
        m = fit_unweighted(make_model(cfg, lp_dim), solver, *data)
        rows.append(dict(seed=seed, lam=np.nan, tau=0, method="Unweighted",
                         regret=solver.mean_regret(m.predict(sp["Z_test"]), sp["y_test"])))
        for lam, tau in itertools.product(args.lambdas, args.taus):
            m = fit_rcif_exp_iterative(make_model(cfg, lp_dim), solver, *data, scheme="directed",
                                       lam=lam, n_iter=tau)
            rows.append(dict(seed=seed, lam=lam, tau=tau, method="Iterative Exp. Directed",
                             regret=solver.mean_regret(m.predict(sp["Z_test"]), sp["y_test"])))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    raw_path = os.path.join(args.results, "sensitivity_raw.csv")
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    df = pd.read_csv(raw_path)

    sub = df[df.method == "Iterative Exp. Directed"]
    g = sub.groupby(["lam", "tau"])["regret"]
    mean_tbl = g.mean().unstack("tau").round(4)
    se_tbl = (g.std() / np.sqrt(g.count())).unstack("tau").round(4)
    base = df[df.method == "Unweighted"].set_index("seed")["regret"]
    print("mean regret:\n", mean_tbl, "\n\nstd err:\n", se_tbl)
    cols = list(mean_tbl.columns)
    with open(os.path.join(args.tables, "sensitivity.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n")
        f.write("$\\lambda \\backslash \\tau$ & " + " & ".join(str(t) for t in cols) + " \\\\\n\\midrule\n")
        for lam in mean_tbl.index:
            f.write(f"{lam} & " + " & ".join(f"{mean_tbl.loc[lam, t]:.4f} $\\pm$ {se_tbl.loc[lam, t]:.4f}"
                                           for t in cols) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nUnweighted: {base.mean():.4f} ± {base.sem():.4f}")
    default = sub[(sub.lam == 0.05) & (sub.tau == 5)].set_index("seed")["regret"]
    if len(default):
        print("\npaired vs default (lambda=0.05, tau=5):")
        for (lam, tau), grp in sub.groupby(["lam", "tau"]):
            if (lam, tau) == (0.05, 5):
                continue
            d = grp.set_index("seed")["regret"].reindex(default.index) - default
            pv = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
            print(f"  lambda={lam:<5} tau={tau:<3} mean diff {d.mean():+.4f}  p={pv:.3g}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].imshow(mean_tbl.values, cmap="viridis_r", aspect="auto")
    axes[0].set_xticks(range(len(mean_tbl.columns)))
    axes[0].set_xticklabels(mean_tbl.columns)
    axes[0].set_yticks(range(len(mean_tbl.index)))
    axes[0].set_yticklabels(mean_tbl.index)
    axes[0].set_xlabel(r"iterations $\tau$")
    axes[0].set_ylabel(r"temperature $\lambda$")
    axes[0].set_title("Mean test regret")
    fig.colorbar(im, ax=axes[0])
    for lam in mean_tbl.index:
        axes[1].errorbar(mean_tbl.columns, mean_tbl.loc[lam].values, yerr=se_tbl.loc[lam].values,
                         marker="o", capsize=3, label=fr"$\lambda={lam}$")
    axes[1].axhline(base.mean(), ls="--", c="gray", label="Unweighted")
    axes[1].set_xlabel(r"iterations $\tau$")
    axes[1].set_ylabel("mean test regret")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "sensitivity_lambda_tau.png"), dpi=200)
    print(f"wrote {raw_path}, {args.tables}/sensitivity.tex and {args.figures}/sensitivity_lambda_tau.png")


if __name__ == "__main__":
    main()
