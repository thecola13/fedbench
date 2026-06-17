"""Plot the differential-privacy privacy/utility trade-off curve.

Uses :class:`PrivacyUtilityVisualizer` (the accuracy-vs-ε figure of the DP-ML
literature, e.g. Abadi et al., CCS 2016) to compare the Laplace and Gaussian
mechanisms added to ``fed_playground``.  For each ε we run a federated
simulation and record the global test MSE, then plot utility against ε on a
log axis with the non-private baseline.

Run:
    uv run python examples/example_privacy_utility_curve.py --save-dir /tmp/figs
    uv run python examples/example_privacy_utility_curve.py   # display interactively
"""

from __future__ import annotations

import argparse

from fed_playground import (
    Environment,
    GaussianDPEncryption,
    LaplaceDPEncryption,
    LinearRegressionModel,
    MeanAggregation,
    NoEncryption,
    PrivacyUtilityVisualizer,
)
from fed_playground.src.dataloader import DataLoader
from fed_playground.src.utils_data import generate_linear_data

EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0]


def _run(scheme):
    X, y = generate_linear_data(n_samples=500, n_features=4, noise=0.05, random_seed=0)
    env = Environment(
        n_parties=5,
        encryption_scheme=scheme,
        aggregation_strategy=MeanAggregation(),
        model_class=LinearRegressionModel,
        model_params={"learning_rate": 0.05, "epochs": 5},
        data_loader=DataLoader(X=X, y=y),
    )
    return env.run_simulation(rounds=8)["global_loss"][-1]


def main() -> None:
    p = argparse.ArgumentParser(description="DP privacy/utility curve demo.")
    p.add_argument("--save-dir", default=None, help="Directory for the PNG; omit to show.")
    args = p.parse_args()

    baseline = _run(NoEncryption())
    # Gaussian's sigma is parameterised here to scale like Laplace's 1/epsilon
    # for a fair visual comparison (both noise levels grow as epsilon shrinks).
    data = {
        "Laplace (ε-DP)": {
            eps: _run(LaplaceDPEncryption(epsilon=eps, sensitivity=1.0, seed=0))
            for eps in EPSILONS
        },
        "Gaussian (≈1/ε noise)": {
            eps: _run(GaussianDPEncryption(sigma=1.0 / eps, seed=0)) for eps in EPSILONS
        },
    }

    print("Privacy/utility (global test MSE; baseline = "
          f"{baseline:.4f}):")
    for name, curve in data.items():
        pretty = "  ".join(f"ε={e}:{curve[e]:.3f}" for e in EPSILONS)
        print(f"  {name:<22} {pretty}")

    viz = PrivacyUtilityVisualizer(save_dir=args.save_dir)
    viz.plot(
        data,
        title="Federated DP: privacy vs utility",
        ylabel="Global test MSE",
        baseline=baseline,
        lower_is_better=True,
    )
    if args.save_dir:
        print(f"\nSaved privacy_utility.png to {args.save_dir}")


if __name__ == "__main__":
    main()
