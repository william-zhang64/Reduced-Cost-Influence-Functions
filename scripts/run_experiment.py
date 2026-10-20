#!/usr/bin/env python
"""Run one experiment configuration; writes <out>/<label>_raw.csv and <out>/<label>_summary.csv.

    python scripts/run_experiment.py table1_shortest_path --additional-methods
    python scripts/run_experiment.py table2_sp_corruption --seed-range 0 5
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rcif import PAPER_CONFIGS, run_experiment, summarize  # noqa: E402
from rcif.experiment import default_methods  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", choices=sorted(PAPER_CONFIGS))
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=None, help="number of seeds (default: config)")
    p.add_argument("--seed-range", type=int, nargs=2, metavar=("START", "STOP"),
                   help="run seeds START..STOP-1")
    p.add_argument("--methods", nargs="+", default=None, help="subset of methods to run")
    p.add_argument("--extra-methods", action="store_true",
                   help="also run the exponential two-sided / directed RCIF variants")
    p.add_argument("--additional-methods", action="store_true",
                   help="also run Clean ERM (oracle), Huber, Trimmed ERM, SPO+ Val Only, SPO+ Finetune")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    base = PAPER_CONFIGS[args.config]
    if args.seeds is not None:
        base = base.__class__(**{**base.to_dict(), "n_seeds": args.seeds})
    os.makedirs(args.out, exist_ok=True)
    for label, cfg in base.expand():
        methods = default_methods(cfg, extra=args.extra_methods, additional=args.additional_methods)
        if args.methods:
            missing = [m for m in args.methods if m not in methods]
            if missing:
                sys.exit(f"unknown methods {missing}; available: {list(methods)}")
            methods = {m: methods[m] for m in args.methods}

        seeds = range(*args.seed_range) if args.seed_range else range(cfg.n_seeds)
        raw_path = os.path.join(args.out, f"{label}_raw.csv")
        prev = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
        done = set(prev.seed.unique()) if len(prev) else set()
        todo = [s for s in seeds if s not in done]

        print(f"=== {label}: {len(todo)} seeds to run ({len(done)} already in {raw_path})")
        print(json.dumps(cfg.to_dict(), default=str))
        df = run_experiment(cfg, seeds=todo, methods=methods, verbose=not args.quiet)
        df = pd.concat([prev, df], ignore_index=True) if len(prev) else df
        df.to_csv(raw_path, index=False)
        table = summarize(df)
        table.to_csv(os.path.join(args.out, f"{label}_summary.csv"))
        print(table[["display", "count"]].to_string(), "\n")


if __name__ == "__main__":
    main()
