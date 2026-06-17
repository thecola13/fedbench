"""Differential privacy and secure aggregation in federated learning.

Two research-grounded encryption schemes added to ``fed_playground``:

* :class:`LaplaceDPEncryption` — the Laplace mechanism for pure ε-differential
  privacy (Dwork, McSherry, Nissim & Smith, TCC 2006).
* :class:`PairwiseMaskingEncryption` — Bonawitz et al.'s secure aggregation
  (ACM CCS 2017): pairwise masks that cancel structurally, so the server only
  ever learns the *sum* of updates, never an individual one.

Part 1 sweeps ε and shows the privacy/utility trade-off of the Laplace
mechanism: tighter privacy (smaller ε) ⇒ more noise ⇒ higher test MSE.
Part 2 shows that secure aggregation is *lossless* — it reproduces the exact
FedAvg result while hiding every individual update in transit.

Run:
    uv run python examples/example_dp_and_secure_agg.py
"""

from __future__ import annotations

import numpy as np

from fed_playground import (
    ClosedFormLinearRegressionModel,
    Environment,
    LaplaceDPEncryption,
    LinearRegressionModel,
    MeanAggregation,
    NoEncryption,
    PairwiseMaskingEncryption,
)
from fed_playground.src.dataloader import DataLoader
from fed_playground.src.utils_data import generate_linear_data

N_PARTIES = 5
ROUNDS = 8


def _data():
    X, y = generate_linear_data(n_samples=500, n_features=4, noise=0.05, random_seed=0)
    return DataLoader(X=X, y=y)


def _run(scheme, model_class=LinearRegressionModel):
    env = Environment(
        n_parties=N_PARTIES,
        encryption_scheme=scheme,
        aggregation_strategy=MeanAggregation(),
        model_class=model_class,
        model_params={"learning_rate": 0.05, "epochs": 5}
        if model_class is LinearRegressionModel
        else {},
        data_loader=_data(),
    )
    return env.run_simulation(rounds=ROUNDS)["global_loss"][-1]


def laplace_privacy_utility() -> None:
    print("=" * 60)
    print("Part 1 — Laplace mechanism: ε privacy/utility trade-off")
    print("=" * 60)
    print("  (smaller ε = stronger privacy = more noise = higher MSE)\n")
    baseline = _run(NoEncryption())
    print(f"  {'no privacy':<18} global test MSE = {baseline:.4f}")
    for eps in (5.0, 1.0, 0.2):
        mse = _run(LaplaceDPEncryption(epsilon=eps, sensitivity=1.0, seed=0))
        print(f"  ε = {eps:<14} global test MSE = {mse:.4f}")
    print()


def secure_aggregation_lossless() -> None:
    print("=" * 60)
    print("Part 2 — Secure aggregation is lossless (vs plaintext FedAvg)")
    print("=" * 60)
    # Closed-form model => deterministic, so the two runs must match to numerical
    # precision: pairwise masks cancel exactly, leaving the FedAvg result.
    plain = _run(NoEncryption(), ClosedFormLinearRegressionModel)
    secure = _run(
        PairwiseMaskingEncryption(n_parties=N_PARTIES, mask_scale=50.0, seed=1),
        ClosedFormLinearRegressionModel,
    )
    print(f"  Plaintext FedAvg            global test MSE = {plain:.6f}")
    print(f"  Secure aggregation (masked) global test MSE = {secure:.6f}")
    print(f"  Identical to 1e-6?           {np.isclose(plain, secure, atol=1e-6)}")
    print(
        "\n  The server saw only masked updates, yet recovered the exact same "
        "model — privacy at zero accuracy cost.\n"
    )


def main() -> None:
    laplace_privacy_utility()
    secure_aggregation_lossless()


if __name__ == "__main__":
    main()
