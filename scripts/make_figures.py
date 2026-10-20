#!/usr/bin/env python
"""Figures 1 and 2 (and the β = 0.2 validation-size figure) from <results>/<label>_raw.csv.

    python scripts/make_figures.py
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


METHODS = ["Iterative Exp. Directed", "Validation Only", "Unweighted"]
FIGURES = {
    "figure1_regret_vs_valsize": ("table3_sp_valsize_beta03__val_size_{}", [5, 10, 20, 50, 100, 150],
                                  "Validation set size", "Regret vs Validation Set Size (β = 0.3)"),
    "figureA_regret_vs_valsize_beta02": ("table3_sp_valsize__val_size_{}", [5, 10, 20, 50, 100, 150, 200],
                                         "Validation set size", "Regret vs Validation Set Size (β = 0.2)"),
    "figure2_regret_vs_corruption": ("table2_sp_corruption__corruption_rate_{}", [0.1, 0.2, 0.3, 0.5],
                                     "Corruption Rate", "Regret vs Corruption Rate"),
}


def load(results_dir, label):
    path = os.path.join(results_dir, f"{label}_raw.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--methods", nargs="+", default=METHODS)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, (pattern, values, xlabel, title) in FIGURES.items():
        frames = [(v, load(args.results, pattern.format(v))) for v in values]
        frames = [(v, f) for v, f in frames if f is not None]
        if not frames:
            print(f"[{name}] no results in {args.results}")
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.array([v for v, _ in frames])
        for m in args.methods:
            stats = np.array([[f[f.method == m].regret.mean(), f[f.method == m].regret.std(ddof=1)]
                              for _, f in frames])
            ax.plot(x, stats[:, 0], marker="o", label=m)
            ax.fill_between(x, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1], alpha=0.15)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Regret", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"wrote {os.path.join(args.out, name)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
