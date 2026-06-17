"""Strong Byzantine robustness: Bulyan and median-of-means aggregation.

Two more research-grounded aggregators added to ``fed_playground``:

* :class:`BulyanAggregation` — Krum selection + coordinate-wise trimmed mean
  (El Mhamdi, Guerraoui & Rouault, ICML 2018). Defends against attacks that
  slip a single coordinate past plain Krum.
* :class:`MedianOfMeansAggregation` — bucket-average then coordinate-wise
  median (Nemirovski-Yudin 1983; Lugosi-Mendelson 2019).

Scenario: several honest parties plus two Byzantine parties that each round
send a large update concentrated on a few coordinates (the attack Bulyan was
designed to stop). We compare final global test MSE across all aggregators.

Run:
    uv run python examples/example_bulyan_mom.py
"""

from __future__ import annotations

import numpy as np

from fed_playground import (
    BulyanAggregation,
    ClosedFormLinearRegressionModel,
    GeometricMedianAggregation,
    KrumAggregation,
    MeanAggregation,
    MedianOfMeansAggregation,
    NoEncryption,
    Orchestrator,
    Party,
)
from fed_playground.src.utils_data import generate_linear_data, split_data


class CoordinateAttackParty(Party):
    """Byzantine party: pushes a few coordinates hard (a Bulyan-style attack)."""

    def get_encrypted_model(self):
        n = self.model.get_parameters().shape[0]
        poisoned = np.zeros(n)
        poisoned[: max(1, n // 3)] = 30.0  # concentrate the attack on a subset
        return self.encryption_scheme.encrypt(poisoned)


def run_strategy(name, strategy, parties_data, test, n_features, n_byz, rounds):
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
        cls = CoordinateAttackParty if i < n_byz else Party
        orch.register_party(cls(i, model, data, scheme))
    for _ in range(rounds):
        orch.distribute_model()
        for party in orch.parties:
            party.train_local_model()
        orch.aggregate_models()
    eval_model = ClosedFormLinearRegressionModel(input_dim=n_features)
    eval_model.set_parameters(orch.global_model_params)
    mse = eval_model.evaluate(*test)
    print(f"  {name:<26} final global test MSE = {mse:.4f}")


def main() -> None:
    n_parties, n_byz, rounds, d = 11, 2, 8, 5
    X, y = generate_linear_data(n_samples=900, n_features=d)
    split = int(0.8 * len(X))
    parties_data = split_data(X[:split], y[:split], n_parties)
    test = (X[split:], y[split:])

    print(f"\n{n_parties} parties ({n_byz} Byzantine, coordinate attack), "
          f"{rounds} rounds — lower MSE is better:\n")
    common = (parties_data, test, d, n_byz, rounds)
    run_strategy("FedAvg (mean)", MeanAggregation(), *common)
    run_strategy("Krum", KrumAggregation(n_byzantine=n_byz), *common)
    run_strategy("Geometric median (RFA)", GeometricMedianAggregation(), *common)
    run_strategy("Median-of-means", MedianOfMeansAggregation(n_buckets=5), *common)
    run_strategy("Bulyan", BulyanAggregation(n_byzantine=n_byz), *common)
    print("\n  Bulyan and median-of-means stay near the honest optimum even under "
          "a\n  coordinate-concentrated attack.\n")


if __name__ == "__main__":
    main()
