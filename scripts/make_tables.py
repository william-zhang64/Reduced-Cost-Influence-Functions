#!/usr/bin/env python
"""Build the paper's tables (Markdown printed, LaTeX written) from <results>/<label>_raw.csv.

    python scripts/make_tables.py --all-methods
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PAPER_ORDER = ["Iterative Exp. Directed", "Exp. Two-Sided", "Directed", "Unweighted",
               "RC Weight", "Validation Only", "SPO Influence Weight"]
EXTRA_ORDER = ["Clean ERM (oracle)", "SPO+ Val Only", "SPO+ Finetune",
               "Exp. Two-Sided (exp)", "Exp. Directed", "Iterative Exp. Two-Sided"]
ROBUST_ORDER = ["Iterative Exp. Directed", "Directed", "Unweighted", "Clean ERM (oracle)",
                "Huber", "Trimmed ERM"]

TABLES = {
    "table1": ("Mean test regret (Table 1)", [
        ("Random LP", "table1_random_lp"),
        ("Shortest Path", "table2_sp_corruption__corruption_rate_0.2"),
        ("NN Upstream", "table1_nn_upstream"),
    ]),
    "table2": ("Shortest path, corruption rate (Table 3)",
               [(f"β = {b}", f"table2_sp_corruption__corruption_rate_{b}") for b in [0.1, 0.2, 0.3, 0.5]]),
    "table3": ("Shortest path, validation size at β = 0.2 (Table 4)",
               [(f"|V| = {v}", f"table3_sp_valsize__val_size_{v}") for v in [5, 10, 20, 50, 100, 150, 200]]),
    "table4": ("Random LP, corruption rate (Table 6)",
               [(f"β = {b}", f"table4_lp_corruption__corruption_rate_{b}") for b in [0.1, 0.2, 0.3, 0.5]]),
    "table5": ("Random LP, validation size (Table 7)",
               [(f"|V| = {v}", f"table5_lp_valsize__val_size_{v}") for v in [5, 10, 20, 50, 100, 150]]),
    "table6": ("Adversarial corruption (Table 8)", [
        ("Random LP", "table6_lp_adversarial"),
        ("Shortest Path", "table6_sp_adversarial"),
    ]),
    "table3_beta03": ("Shortest path, validation size at β = 0.3 (Table 5)",
                      [(f"|V| = {v}", f"table3_sp_valsize_beta03__val_size_{v}") for v in [5, 10, 20, 50, 100, 150]]),
    "misspecification": ("Misspecification, PyEPO polynomial degree (Table 2)", [
        (f"deg = {d}", f"misspecification__deg_{d}") for d in [1, 2, 4]
    ]),
    "robust_regression": ("Robust regression under Gaussian and directional corruption (Table 9)", [
        ("Random LP, Gaussian β = 0.3", "table1_random_lp"),
        ("Shortest Path, Gaussian β = 0.2", "robust_gaussian"),
        ("Shortest Path, Directional β = 0.4", "directional"),
    ]),
}


def load_column(results_dir, label):
    """{method: (mean, ci95, n)} for one label, or None if absent."""
    path = os.path.join(results_dir, f"{label}_raw.csv")
    if not os.path.exists(path):
        return None
    g = pd.read_csv(path).groupby("method")["regret"].agg(["mean", "std", "count"])
    return {m: (r["mean"], 1.96 * r["std"] / np.sqrt(r["count"]), int(r["count"]))
            for m, r in g.iterrows()}


def render(title, columns, results_dir, methods):
    cols = [(name, load_column(results_dir, label)) for name, label in columns]
    cols = [(n, c) for n, c in cols if c is not None]
    if not cols:
        return None, None
    best = {n: min(c[m][0] for m in methods if m in c and m != "Clean ERM (oracle)")
            for n, c in cols if any(m in c and m != "Clean ERM (oracle)" for m in methods)}
    md = [f"### {title}", "", "| Method | " + " | ".join(n for n, _ in cols) + " |",
          "|---|" + "---|" * len(cols)]
    tex = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
           "Method & " + " & ".join(n.replace("β", "$\\beta$").replace("α", "$\\alpha$") for n, _ in cols) + " \\\\", "\\midrule"]
    for m in methods:
        if not any(m in c for _, c in cols):
            continue
        md_cells, tex_cells = [], []
        for n, c in cols:
            if m not in c:
                md_cells.append("–"); tex_cells.append("--"); continue
            mean, ci, _ = c[m]
            s = f"{mean:.4f} ± {ci:.4f}"
            is_best = abs(mean - best[n]) < 1e-12
            md_cells.append(f"**{s}**" if is_best else s)
            tex_cells.append(f"\\textbf{{{mean:.4f} $\\pm$ {ci:.4f}}}" if is_best else f"{mean:.4f} $\\pm$ {ci:.4f}")
        md.append(f"| {m} | " + " | ".join(md_cells) + " |")
        tex.append(f"{m} & " + " & ".join(tex_cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    n_seeds = sorted({c[m][2] for _, c in cols for m in c})
    md.append(f"\nSeeds per column: {n_seeds}\n")
    for n, c in cols:
        if "Clean ERM (oracle)" in c and "Unweighted" in c:
            dmg = c["Unweighted"][0] - c["Clean ERM (oracle)"][0]
            parts = [f"{m}: {(c['Unweighted'][0] - c[m][0]) / dmg:.0%}" for m in methods
                     if m in c and m not in ("Unweighted", "Clean ERM (oracle)")]
            md.append(f"{n}: corruption damage (Unweighted − Clean ERM) = {dmg:.4f}; "
                      f"fraction recovered — " + ", ".join(parts) + "\n")
    return "\n".join(md), "\n".join(tex)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="tables")
    p.add_argument("--all-methods", action="store_true", help="include the additional baselines")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for key, (title, columns) in TABLES.items():
        if key == "robust_regression":
            methods = ROBUST_ORDER
        elif key == "misspecification":
            methods = PAPER_ORDER + EXTRA_ORDER[:3]
        else:
            methods = PAPER_ORDER + (EXTRA_ORDER if args.all_methods else [])
        md, tex = render(title, columns, args.results, methods)
        if md is None:
            print(f"[{key}] no results found in {args.results}")
            continue
        print(md)
        with open(os.path.join(args.out, f"{key}.tex"), "w", encoding="utf-8") as f:
            f.write(tex + "\n")


if __name__ == "__main__":
    main()
