"""Elastic-net and Poisson GLM models in a federated setting.

Two research-grounded models added to ``fed_playground``:

* :class:`ElasticNetRegressionModel` — blended L1/L2 penalty (Zou & Hastie,
  JRSS-B 2005): sparse like LASSO but stable on correlated features.
* :class:`PoissonRegressionModel` — log-linear GLM for count data
  (Nelder & Wedderburn, JRSS-A 1972): predicts non-negative rates.

Part 1 contrasts the elastic-net regularization path against pure LASSO on a
problem with correlated predictors.
Part 2 federates a Poisson regression over synthetic count data and reports
the mean Poisson deviance (lower is better).

Run:
    uv run python examples/example_elasticnet_poisson.py
"""

from __future__ import annotations

import numpy as np

from fed_playground import (
    ElasticNetRegressionModel,
    Environment,
    LassoRegressionModel,
    MeanAggregation,
    NoEncryption,
    PoissonRegressionModel,
)
from fed_playground.src.dataloader import DataLoader


def elasticnet_vs_lasso() -> None:
    print("=" * 64)
    print("Part 1 — Elastic net vs LASSO on correlated features")
    print("=" * 64)
    rng = np.random.default_rng(0)
    n = 400
    base = rng.standard_normal(n)
    # Features 0,1,2 are correlated copies of the same signal; 3-5 are noise.
    X = np.column_stack([
        base + 0.01 * rng.standard_normal(n),
        base + 0.01 * rng.standard_normal(n),
        base + 0.01 * rng.standard_normal(n),
        rng.standard_normal(n),
        rng.standard_normal(n),
        rng.standard_normal(n),
    ])
    y = 1.5 * base + 0.05 * rng.standard_normal(n)

    lasso = LassoRegressionModel(input_dim=6, alpha=0.2)
    lasso.train(X, y)
    enet = ElasticNetRegressionModel(input_dim=6, alpha=0.2, l1_ratio=0.5)
    enet.train(X, y)

    def fmt(w):
        return np.array2string(w, precision=2, suppress_small=True)

    print(f"  LASSO weights        {fmt(lasso.weights)}")
    print(f"  Elastic-net weights  {fmt(enet.weights)}")
    print(
        "\n  LASSO tends to pick ONE of the correlated copies (0-2) and zero the\n"
        "  rest; elastic net spreads weight across the correlated group (the\n"
        "  'grouping effect'). Both zero the noise features 3-5.\n"
    )


def federated_poisson() -> None:
    print("=" * 64)
    print("Part 2 — Federated Poisson regression (count data)")
    print("=" * 64)
    rng = np.random.default_rng(1)
    n, d = 1000, 3
    X = 0.5 * rng.standard_normal((n, d))
    true_w = np.array([0.8, -0.5, 0.3])
    y = rng.poisson(np.exp(X @ true_w + 0.2)).astype(float)

    env = Environment(
        n_parties=5,
        encryption_scheme=NoEncryption(),
        aggregation_strategy=MeanAggregation(),
        model_class=PoissonRegressionModel,
        model_params={"learning_rate": 0.05, "epochs": 100},
        data_loader=DataLoader(X=X, y=y),
    )
    history = env.run_simulation(rounds=10)
    print(f"  Final global mean Poisson deviance: {history['global_loss'][-1]:.4f}")
    print(f"  (Empirical mean count = {y.mean():.3f}; model predicts rates >= 0)\n")


def main() -> None:
    elasticnet_vs_lasso()
    federated_poisson()


if __name__ == "__main__":
    main()
