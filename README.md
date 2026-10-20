# Reduced Cost Influence Functions (RCIF) for Predict-then-Optimize under Noisy Data

Code for the paper *Reduced Cost Influence Functions for Predict-then-Optimize
under Noisy Data*. It implements RCIF scoring and reweighting, all baselines of
the paper, the random-LP and shortest-path benchmarks, and the scripts that
produce every table and figure.

```
rcif/                 library
  lp.py               Gurobi LP solver, optimal basis, reduced costs, edge directions, regret
  models.py           linear model (closed-form influence) and two-layer NN (Gauss-Newton influence)
  scoring.py          RCIF scores (two-sided / directed), RC-margin and SPO+-influence scores
  methods.py          the seven methods of the paper + registries
  baselines.py        clean-ERM oracle, Huber, trimmed ERM, SPO+ val-only / fine-tune
  data.py             random LP instances, PyEPO shortest path, Gaussian / cone-flip / directional corruption
  experiment.py       per-seed runner, summaries, paired Wilcoxon tests
  configs.py          one configuration per table and ablation
scripts/
  run_experiment.py   run a configuration, writes results/<label>_raw.csv and _summary.csv
  make_tables.py      all tables as Markdown and LaTeX
  make_figures.py     Figures 1 and 2
  paired_tests.py     paired significance tests for Table 1
  run_sensitivity.py  lambda x tau sensitivity grid
results/              per-seed results of every experiment in the paper
tables/, figures/     built from results/ by the scripts above
tests/                correctness tests
```

## Installation

Requires Python 3.11, [Gurobi](https://www.gurobi.com) with a licence
(free academic licences are available) and [PyEPO](https://github.com/khalil-research/PyEPO)
for the shortest-path benchmark.

```bash
conda env create -f environment.yml && conda activate rcif
# or: pip install -r requirements.txt
python -m pytest tests -q
```

## Reproducing the paper

Each table has a configuration in `rcif/configs.py`; running one writes
`results/<label>_raw.csv` (method, seed, test regret) and a summary.

| config | table | seeds |
|---|---|---|
| `table1_random_lp` | Table 1, Random LP | 100 |
| `table1_shortest_path` | Table 1, Shortest Path (= Table 3, β = 0.2) | 50 |
| `table1_nn_upstream` | Table 1, NN Upstream | 50 |
| `table2_sp_corruption` | Table 3, Figure 2 | 50 per β |
| `table3_sp_valsize` | Table 4 (β = 0.2, \|V\| up to 200) | 100 per \|V\| |
| `table3_sp_valsize_beta03` | Table 5, Figure 1 (β = 0.3) | 100 per \|V\| |
| `table4_lp_corruption` | Table 6 | 100 per β |
| `table5_lp_valsize` | Table 7 | 100 per \|V\| |
| `table6_lp_adversarial` | Table 8, Random LP | 100 |
| `table6_sp_adversarial` | Table 8, Shortest Path | 50 |
| `robust_gaussian`, `directional` | Table 9 (Gaussian / directional corruption) | 50 |
| `misspecification` | Table 2 (PyEPO degree sweep) | 50 per degree |

```bash
python scripts/run_experiment.py table1_shortest_path --additional-methods
python scripts/run_experiment.py table3_sp_valsize --additional-methods     # all seven |V| values
python scripts/run_sensitivity.py --lambdas 0.01 0.05 0.1 0.2 0.5 --taus 3 5 10 --seed-range 50 100   # held-out seeds
python scripts/make_tables.py --all-methods
python scripts/paired_tests.py
python scripts/make_figures.py
```

`--additional-methods` adds the Clean-ERM oracle, Huber, trimmed ERM and the
two SPO+ baselines to the seven methods of the paper (Huber and the SPO+
baselines are defined for the linear upstream model). Runs are resumable
(seeds already present in the raw file are skipped); `--seed-range 0 5` and
`--methods Unweighted "Validation Only"` give quick checks. The shipped `results/`
directory holds the per-seed results behind the paper, so the three
table/figure scripts can be run directly.

### Method names

| paper | code (`rcif.methods`) |
|---|---|
| Iterative Exp. Directed | `fit_rcif_exp_iterative(scheme="directed", lam=0.05, n_iter=5)` |
| Exp. Two-Sided | `fit_rcif_additive(scheme="two_sided", phi=0.05)` |
| Directed | `fit_rcif_additive(scheme="directed", phi=0.05)` |
| Unweighted | `fit_unweighted` |
| RC Weight | `fit_rc_weight(lam=0.05)` |
| Validation Only | `fit_validation_only` |
| SPO Influence Weight | `fit_spo_influence(lr=0.01, n_iter=5)` |
| Clean ERM (oracle), Huber, Trimmed ERM, SPO+ Val Only, SPO+ Finetune | `rcif.baselines` |

`--extra-methods` adds the single-round exponential two-sided / directed
variants and iterative exponential two-sided.

## Library use

```python
from rcif import LPSolver, LinearModel, rcif_scores
from rcif.data import make_random_lp, generate_linear_data
import numpy as np

A, b, theta = make_random_lp(num_constraints=20, lp_dim=10, input_dim=20)
solver = LPSolver(A, b)
rng = np.random.default_rng(0)
Z_tr, C_tr = generate_linear_data(200, theta, 0.5, rng)
Z_val, C_val = generate_linear_data(50, theta, 0.5, rng)

model = LinearModel().fit(Z_tr, C_tr)
psi = rcif_scores(model, solver, Z_tr, C_tr, Z_val, C_val, scheme="directed")  # Ψ^D_j
weights = np.exp(0.05 * psi); weights *= len(psi) / weights.sum()
model.fit(Z_tr, C_tr, sample_weight=weights)
```

`LPSolver.solve_with_basis(c)` returns the optimal vertex together with
reduced costs, nonbasic mask and edge directions δ(B, k); `LPSolution.margins`,
`fragile_direction`, `entering_direction` and `same_cone` implement the cone
geometry of Section 4.
