#!/usr/bin/env python
"""Paired Wilcoxon signed-rank tests over seeds, against --baseline and --ours, for each label.

    python scripts/paired_tests.py
    python scripts/paired_tests.py --labels table3_sp_valsize__val_size_50
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ORDER = ["Iterative Exp. Directed", "Exp. Two-Sided", "Directed", "Unweighted", "RC Weight",
         "Validation Only", "SPO Influence Weight", "Clean ERM (oracle)", "Huber", "Trimmed ERM",
         "SPO+ Val Only", "SPO+ Finetune", "Exp. Two-Sided (exp)", "Exp. Directed",
         "Iterative Exp. Two-Sided"]


def paired(df, ref):
    base = df[df.method == ref].set_index("seed")["regret"]
    rows = []
    for m in [m for m in ORDER if m in set(df.method)]:
        x = df[df.method == m].set_index("seed")["regret"].reindex(base.index)
        d = (x - base).dropna()
        if m == ref or len(d) == 0:
            rows.append(dict(method=m, mean=x.mean(), diff=np.nan, win=np.nan, p=np.nan))
            continue
        p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
        rows.append(dict(method=m, mean=x.mean(), diff=d.mean(), win=(d < 0).mean(), p=p))
    return pd.DataFrame(rows)


def fmt_p(p):
    return "" if np.isnan(p) else ("< 0.001" if p < 1e-3 else f"{p:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--baseline", default="Unweighted")
    ap.add_argument("--ours", default="Iterative Exp. Directed")
    ap.add_argument("--exclude", nargs="*", default=["Huber", "Trimmed ERM"])
    ap.add_argument("--labels", nargs="+", default=["table1_random_lp",
                    "table2_sp_corruption__corruption_rate_0.2", "table1_nn_upstream"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    md = []
    for label in args.labels:
        path = os.path.join(args.results, f"{label}_raw.csv")
        if not os.path.exists(path):
            print(f"[{label}] no raw file in {args.results}")
            continue
        df = pd.read_csv(path)
        df = df[~df.method.isin(args.exclude)]
        if args.baseline not in set(df.method):
            continue
        tb = paired(df, args.baseline)
        to = paired(df, args.ours) if args.ours in set(df.method) else None
        n = df.seed.nunique()
        md.append(f"### {label} ({n} seeds; paired Wilcoxon signed-rank over seeds)\n")
        hdr = f"| Method | mean regret | Δ vs {args.baseline} | wins vs {args.baseline} | p |"
        tex = ["\\begin{tabular}{lrrrr" + ("rrr" if to is not None else "") + "}", "\\toprule",
               f"Method & regret & $\\Delta$ vs {args.baseline} & wins & $p$"
               + (f" & $\\Delta$ vs ours & wins & $p$" if to is not None else "") + " \\\\", "\\midrule"]
        if to is not None:
            hdr += f" Δ vs {args.ours} | wins vs {args.ours} | p |"
        md += [hdr, "|---" * (5 if to is None else 8) + "|"]
        for i, r in tb.iterrows():
            cells = [r["method"], f"{r['mean']:.4f}",
                     "" if np.isnan(r["diff"]) else f"{r['diff']:+.4f}",
                     "" if np.isnan(r["win"]) else f"{r['win']:.0%}", fmt_p(r["p"])]
            if to is not None:
                ro = to[to.method == r["method"]].iloc[0]
                cells += ["" if np.isnan(ro["diff"]) else f"{ro['diff']:+.4f}",
                          "" if np.isnan(ro["win"]) else f"{ro['win']:.0%}", fmt_p(ro["p"])]
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(c.replace("%", "\\%").replace("Δ", "$\\Delta$") for c in cells) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        md.append("")
        with open(os.path.join(args.out, f"paired_{label}.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
