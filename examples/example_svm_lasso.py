"""Federated training with the new SVM and LASSO models.

Two research-grounded models added to ``fed_playground``:

* :class:`SVMModel` — linear SVM via Pegasos (Shalev-Shwartz et al.,
  Mathematical Programming 2011): hinge loss + L2, sub-gradient steps 1/(λt).
* :class:`LassoRegressionModel` — L1 regression via coordinate descent
  (Tibshirani 1996; Friedman, Hastie & Tibshirani 2010): sparse, selects
  features by driving irrelevant weights to exactly zero.

Part 1 federates a linear SVM over a separable classification task.
Part 2 fits LASSO on data where only 2 of 8 features matter and prints the
recovered (sparse) weight vector.

Run:
    uv run python examples/example_svm_lasso.py
"""

from __future__ import annotations

import numpy as np

from fed_playground import (
    Environment,
    LassoRegressionModel,
    MeanAggregation,
    NoEncryption,
    SVMModel,
)
from fed_playground.src.dataloader import DataLoader


def federated_svm() -> None:
    print("=" * 60)
    print("Part 1 — Federated linear SVM (Pegasos)")
    print("=" * 60)
    rng = np.random.default_rng(0)
    n, d = 400, 5
    pos = rng.standard_normal((n // 2, d)) + 2.0
    neg = rng.standard_normal((n // 2, d)) - 2.0
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    env = Environment(
        n_parties=5,
        encryption_scheme=NoEncryption(),
        aggregation_strategy=MeanAggregation(),
        model_class=SVMModel,
        model_params={"lambda_reg": 0.01, "epochs": 15},
        data_loader=DataLoader(X=X, y=y),
    )
    history = env.run_simulation(rounds=8)
    print(f"  Final global accuracy: {history['party_loss'][-1]:.3f} "
          "(party-average; SVM.evaluate returns accuracy)\n")


def federated_lasso() -> None:
    print("=" * 60)
    print("Part 2 — LASSO feature selection")
    print("=" * 60)
    rng = np.random.default_rng(1)
    n, d = 500, 8
    X = rng.standard_normal((n, d))
    true_w = np.array([4.0, -3.0, 0, 0, 0, 0, 0, 0])  # only features 0,1 matter
    y = X @ true_w + 0.05 * rng.standard_normal(n)

    for alpha in (0.0, 0.1, 0.5):
        model = LassoRegressionModel(input_dim=d, alpha=alpha)
        model.train(X, y)
        nnz = int(np.sum(np.abs(model.weights) > 1e-8))
        weights = np.array2string(model.weights, precision=2, suppress_small=True)
        print(f"  alpha={alpha:<4}  non-zero weights={nnz}  w={weights}")
    print("\n  True support is features 0 and 1; larger alpha → sparser fit.\n")


def main() -> None:
    federated_svm()
    federated_lasso()


if __name__ == "__main__":
    main()
