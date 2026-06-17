"""Byzantine-robust federated aggregation under a poisoning attack.

Demonstrates the two research-grounded aggregators added to ``fed_playground``:

* :class:`KrumAggregation` — Blanchard et al., "Machine Learning with
  Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017.
* :class:`GeometricMedianAggregation` — Pillutla et al., "Robust Aggregation
  for Federated Learning" (RFA), IEEE TSP 2022.

Scenario: a handful of honest parties hold slices of a clean linear-regression
problem; one *Byzantine* party sends a wildly corrupted update every round.
FedAvg (the mean) is dragged off by the attacker, while Krum and the geometric
median ignore it.  We report the final global test MSE for each strategy.

Run:
    uv run python examples/example_byzantine_robust.py
    uv run python examples/example_byzantine_robust.py --parties 8 --rounds 15
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from fed_playground import (
    CenteredClippingAggregation,
    ClosedFormLinearRegressionModel,
    GeometricMedianAggregation,
    KrumAggregation,
    MeanAggregation,
    NoEncryption,
    Orchestrator,
    Party,
)
from fed_playground.src.utils_data import generate_linear_data, split_data

logging.basicConfig(level=logging.WARNING, format="%(message)s")


class ByzantineParty(Party):
    """A malicious party that ignores its data and emits a huge fixed update."""

    def __init__(self, party_id, model, data, encryption_scheme, magnitude=50.0):
        super().__init__(party_id, model, data, encryption_scheme)
        self.magnitude = magnitude

    def get_encrypted_model(self):
        # Sign-flipped, scaled garbage of the right shape — a classic FL attack.
        n = self.model.get_parameters().shape[0]
        poisoned = self.magnitude * np.ones(n)
        poisoned[::2] *= -1.0
        return self.encryption_scheme.encrypt(poisoned)


def _build_orchestrator(strategy, parties_data, n_features, n_byzantine):
    scheme = NoEncryption()
    orch = Orchestrator(
        aggregation_strategy=strategy,
        encryption_scheme=scheme,
        initial_model_params=ClosedFormLinearRegressionModel(
            input_dim=n_features
        ).get_parameters(),
    )
    for i, data in enumerate(parties_data):
        model = ClosedFormLinearRegressionModel(input_dim=n_features)
        cls = ByzantineParty if i < n_byzantine else Party
        orch.register_party(cls(i, model, data, scheme))
    return orch


def run_strategy(name, strategy, parties_data, test, n_features, n_byzantine, rounds):
    orch = _build_orchestrator(strategy, parties_data, n_features, n_byzantine)
    X_test, y_test = test
    for _ in range(rounds):
        orch.distribute_model()
        for party in orch.parties:
            party.train_local_model()
        orch.aggregate_models()
    eval_model = ClosedFormLinearRegressionModel(input_dim=n_features)
    eval_model.set_parameters(orch.global_model_params)
    mse = eval_model.evaluate(X_test, y_test)
    print(f"  {name:<28} final global test MSE = {mse:.4f}")
    return mse


def main() -> None:
    p = argparse.ArgumentParser(description="Byzantine-robust FL aggregation demo.")
    p.add_argument("--parties", type=int, default=7, help="Total parties.")
    p.add_argument("--byzantine", type=int, default=1, help="Malicious parties.")
    p.add_argument("--rounds", type=int, default=10, help="Federated rounds.")
    p.add_argument("--features", type=int, default=5, help="Input features.")
    args = p.parse_args()

    X, y = generate_linear_data(n_samples=600, n_features=args.features)
    split = int(0.8 * len(X))
    (X_train, y_train), test = (X[:split], y[:split]), (X[split:], y[split:])
    parties_data = split_data(X_train, y_train, args.parties)

    print(
        f"\n{args.parties} parties ({args.byzantine} Byzantine), "
        f"{args.rounds} rounds — lower MSE is better:\n"
    )
    common = (parties_data, test, args.features, args.byzantine, args.rounds)
    run_strategy("FedAvg (mean)", MeanAggregation(), *common)
    run_strategy("Krum", KrumAggregation(n_byzantine=args.byzantine), *common)
    run_strategy(
        "Multi-Krum (m=parties-byz-2)",
        KrumAggregation(
            n_byzantine=args.byzantine,
            n_selected=max(1, args.parties - args.byzantine - 2),
        ),
        *common,
    )
    run_strategy("Geometric median (RFA)", GeometricMedianAggregation(), *common)
    run_strategy(
        "Centered clipping", CenteredClippingAggregation(clip_radius=1.0, n_iters=5), *common
    )
    print(
        "\nFedAvg is corrupted by the attacker; Krum and the geometric median "
        "stay close to the honest optimum.\n"
    )


if __name__ == "__main__":
    main()
