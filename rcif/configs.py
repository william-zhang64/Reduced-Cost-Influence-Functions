"""Experiment configurations, one per table and figure."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass
class ExperimentConfig:
    name: str
    problem: str                      # "random_lp" | "shortest_path"
    model: str = "linear"             # "linear" | "nn"
    train_size: int = 200
    val_size: int = 50
    test_size: int = 150
    corruption_rate: float = 0.3
    corruption: str = "gaussian"      # "gaussian" | "cone_flip" | "directional"
    noise_scale: float = 2.0
    budget: float = 0.3
    alpha: float = 1.5
    n_seeds: int = 50
    input_dim: int = 20
    lp_dim: int = 10
    num_constraints: int = 20
    instance_seed: int = 42
    label_noise: float = 0.5
    grid: tuple = (4, 4)
    deg: int = 1
    noise_width: float = 0.5
    hidden_size: int = 8
    nn_lr: float = 1e-3
    nn_max_iter: int = 500
    sweep_param: str | None = None
    sweep_values: list = field(default_factory=list)
    paper_ref: str = ""

    @property
    def n_total(self) -> int:
        return self.train_size + self.val_size + self.test_size

    def expand(self):
        """(label, config) pairs, one per sweep value."""
        if self.sweep_param is None:
            yield self.name, self
            return
        for v in self.sweep_values:
            yield f"{self.name}__{self.sweep_param}_{v}", replace(
                self, sweep_param=None, sweep_values=[], **{self.sweep_param: v})

    def to_dict(self):
        return asdict(self)


PAPER_CONFIGS = {
    "table1_random_lp": ExperimentConfig(
        name="table1_random_lp", problem="random_lp", input_dim=20, lp_dim=10,
        num_constraints=20, train_size=200, val_size=50, test_size=150,
        corruption_rate=0.3, n_seeds=100,
        paper_ref="Table 1, column 'Random LP'",
    ),
    "table1_shortest_path": ExperimentConfig(
        name="table1_shortest_path", problem="shortest_path", grid=(4, 4),
        input_dim=30, train_size=400, val_size=100, test_size=200,
        corruption_rate=0.2, n_seeds=50,
        paper_ref="Table 1, column 'Shortest Path' (= Table 3, beta = 0.2)",
    ),
    "table1_nn_upstream": ExperimentConfig(
        name="table1_nn_upstream", problem="shortest_path", model="nn", grid=(5, 4),
        input_dim=30, train_size=200, val_size=100, test_size=200,
        corruption_rate=0.3, n_seeds=50,
        paper_ref="Table 1, column 'NN Upstream'",
    ),
    "table2_sp_corruption": ExperimentConfig(
        name="table2_sp_corruption", problem="shortest_path", grid=(4, 4),
        input_dim=30, train_size=400, val_size=100, test_size=200, n_seeds=50,
        sweep_param="corruption_rate", sweep_values=[0.1, 0.2, 0.3, 0.5],
        paper_ref="Table 3 and Figure 2",
    ),
    "table3_sp_valsize": ExperimentConfig(
        name="table3_sp_valsize", problem="shortest_path", grid=(4, 4),
        input_dim=20, train_size=500, test_size=200, corruption_rate=0.2, n_seeds=100,
        sweep_param="val_size", sweep_values=[5, 10, 20, 50, 100, 150, 200],
        paper_ref="Table 4",
    ),
    "table3_sp_valsize_beta03": ExperimentConfig(
        name="table3_sp_valsize_beta03", problem="shortest_path", grid=(4, 4),
        input_dim=20, train_size=500, test_size=200, corruption_rate=0.3, n_seeds=100,
        sweep_param="val_size", sweep_values=[5, 10, 20, 50, 100, 150],
        paper_ref="Table 5 and Figure 1",
    ),
    "table4_lp_corruption": ExperimentConfig(
        name="table4_lp_corruption", problem="random_lp", input_dim=10, lp_dim=10,
        num_constraints=20, train_size=200, val_size=50, test_size=100, n_seeds=100,
        sweep_param="corruption_rate", sweep_values=[0.1, 0.2, 0.3, 0.5],
        paper_ref="Table 6",
    ),
    "table5_lp_valsize": ExperimentConfig(
        name="table5_lp_valsize", problem="random_lp", input_dim=10, lp_dim=10,
        num_constraints=20, train_size=200, test_size=100, corruption_rate=0.5, n_seeds=100,
        sweep_param="val_size", sweep_values=[5, 10, 20, 50, 100, 150],
        paper_ref="Table 7",
    ),
    "table6_lp_adversarial": ExperimentConfig(
        name="table6_lp_adversarial", problem="random_lp", input_dim=10, lp_dim=10,
        num_constraints=20, train_size=200, val_size=50, test_size=100,
        corruption_rate=0.3, corruption="cone_flip", budget=0.3, n_seeds=100,
        paper_ref="Table 8, column 'Random LP'",
    ),
    "table6_sp_adversarial": ExperimentConfig(
        name="table6_sp_adversarial", problem="shortest_path", grid=(5, 5),
        input_dim=30, train_size=300, val_size=100, test_size=200,
        corruption_rate=0.3, corruption="cone_flip", budget=0.05, n_seeds=50,
        paper_ref="Table 8, column 'Shortest Path'",
    ),
    "robust_gaussian": ExperimentConfig(
        name="robust_gaussian", problem="shortest_path", grid=(4, 4),
        input_dim=20, train_size=400, val_size=100, test_size=200,
        corruption_rate=0.2, n_seeds=50,
        paper_ref="Table 9, Gaussian corruption",
    ),
    "directional": ExperimentConfig(
        name="directional", problem="shortest_path", grid=(4, 4),
        input_dim=20, train_size=400, val_size=100, test_size=200,
        corruption_rate=0.4, corruption="directional", alpha=1.5, n_seeds=50,
        paper_ref="Table 9, directional corruption",
    ),
    "misspecification": ExperimentConfig(
        name="misspecification", problem="shortest_path", grid=(4, 4),
        input_dim=30, train_size=400, val_size=100, test_size=200,
        corruption_rate=0.2, n_seeds=50,
        sweep_param="deg", sweep_values=[1, 2, 4],
        paper_ref="Table 2",
    ),
}
