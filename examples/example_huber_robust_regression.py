"""Robust regression with the Huber loss under corrupted targets.

:class:`HuberRegressionModel` (Huber, Annals of Math. Stat. 1964) caps the
influence of large residuals, so a few grossly wrong target values cannot drag
the fit the way they do under ordinary least squares.  This is a robustness
axis orthogonal to the L1/L2 weight penalties (Lasso/ElasticNet) and to the
robust aggregators (Krum/Bulyan): here the *loss function* itself is robust.

We corrupt 10% of the training targets and compare, against the CLEAN targets,
the test error of OLS vs Huber.

Run:
    uv run python examples/example_huber_robust_regression.py
"""

from __future__ import annotations

import numpy as np

from fed_playground import (
    ClosedFormLinearRegressionModel,
    Environment,
    HuberRegressionModel,
    MeanAggregation,
    NoEncryption,
)
from fed_playground.src.dataloader import DataLoader
from fed_playground.src.utils_data import generate_linear_data


def main() -> None:
    X, y_clean = generate_linear_data(n_samples=600, n_features=4, noise=0.05, random_seed=0)
    rng = np.random.default_rng(0)
    y_corrupt = y_clean.copy()
    idx = rng.choice(len(y_clean), size=len(y_clean) // 10, replace=False)
    y_corrupt[idx] += 100.0  # 10% gross outliers in the target

    split = int(0.8 * len(X))
    X_test, y_test = X[split:], y_clean[split:]  # evaluate on clean targets

    print("\nTraining on targets with 10% gross outliers; evaluating on clean targets.")
    print("Lower test MSE is better:\n")

    for name, model_cls, params in [
        ("OLS (least squares)", ClosedFormLinearRegressionModel, {}),
        ("Huber (delta=1.0)", HuberRegressionModel,
         {"delta": 1.0, "learning_rate": 0.05, "epochs": 300}),
    ]:
        env = Environment(
            n_parties=4,
            encryption_scheme=NoEncryption(),
            aggregation_strategy=MeanAggregation(),
            model_class=model_cls,
            model_params=params,
            data_loader=DataLoader(X=X[:split], y=y_corrupt[:split]),
        )
        env.run_simulation(rounds=6)
        eval_model = model_cls(input_dim=4, **params)
        eval_model.set_parameters(env.orchestrator.global_model_params)
        mse = float(np.mean((eval_model.predict(X_test) - y_test) ** 2))
        print(f"  {name:<22} test MSE (vs clean) = {mse:.4f}")

    print("\n  Huber's linear tail caps each outlier's influence, so it stays "
          "close to\n  the clean optimum while OLS is dragged off.\n")


if __name__ == "__main__":
    main()
