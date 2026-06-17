"""Byzantine-robust aggregation strategies.

Compares MeanAggregation, TrimmedMeanAggregation, and MedianAggregation
in the presence of simulated Byzantine (corrupted) parties.

The experiment runs two scenarios:

    Honest:   all parties receive clean data.
    Byzantine: a fraction of parties have their updates replaced with
               large random noise before the orchestrator aggregates —
               simulating malicious gradient poisoning.

Usage
-----
    uv run python examples/example_robust_aggregation.py
"""

import numpy as np

from fed_playground import (
    ClosedFormLinearRegressionModel,
    Environment,
    MeanAggregation,
    MedianAggregation,
    NoEncryption,
    Orchestrator,
    Party,
    TrimmedMeanAggregation,
    TrainingHistoryVisualizer,
)
from fed_playground.src.utils_data import generate_linear_data, split_data

ROUNDS = 20
N_PARTIES = 10
N_FEATURES = 5
N_SAMPLES = 600
N_BYZANTINE = 5   # number of corrupted parties
SEED = 13


def run_scenario(
    aggregation_strategy,
    byzantine: bool,
    label: str,
) -> list[float]:
    """Run a federated simulation, optionally injecting Byzantine noise.

    Args:
        aggregation_strategy: The aggregation strategy instance to use.
        byzantine: If True, N_BYZANTINE parties submit random parameters.
        label: Display name for this run.

    Returns:
        List of global test MSE values, one per round.
    """
    rng = np.random.default_rng(SEED)
    X, y = generate_linear_data(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_seed=SEED
    )
    party_data = split_data(X, y, n_parties=N_PARTIES, random_seed=SEED)

    # Hold out 20 % of the data as a global test set.
    n_test = N_SAMPLES // 5
    X_test, y_test = X[:n_test], y[:n_test]

    scheme = NoEncryption()
    parties = []
    for i, (X_i, y_i) in enumerate(party_data):
        m = ClosedFormLinearRegressionModel(input_dim=N_FEATURES)
        p = Party(party_id=i, model=m, data=(X_i, y_i), encryption_scheme=scheme)
        parties.append(p)

    # Use a dedicated eval model to assess the global params on the test set.
    eval_model = ClosedFormLinearRegressionModel(input_dim=N_FEATURES)
    orch = Orchestrator(
        aggregation_strategy=aggregation_strategy,
        encryption_scheme=scheme,
    )
    for p in parties:
        orch.register_party(p)

    global_losses: list[float] = []

    for _ in range(ROUNDS):
        orch.distribute_model()
        for p in parties:
            p.train_local_model()

        if byzantine:
            # Replace the first N_BYZANTINE parties' models with random garbage.
            for p in parties[:N_BYZANTINE]:
                noise = rng.normal(0, 100.0, size=N_FEATURES + 1)
                p.model.set_parameters(noise)

        orch.aggregate_models()
        eval_model.set_parameters(orch.global_model_params)
        loss = eval_model.evaluate(X_test, y_test)
        global_losses.append(loss)

    return global_losses


strategies = {
    "FedAvg (mean)": MeanAggregation(),
    "Trimmed mean (10 %)": TrimmedMeanAggregation(trim_fraction=0.1),
    "Coordinate median": MedianAggregation(),
}

print("=" * 60)
print("HONEST setting (no Byzantine parties)")
print("=" * 60)
honest_histories: dict[str, list[float]] = {}
for name, strat in strategies.items():
    losses = run_scenario(strat, byzantine=False, label=name)
    honest_histories[name] = losses
    print(f"  {name:<28}  final MSE = {losses[-1]:.5f}")

print()
print("=" * 60)
print(f"BYZANTINE setting ({N_BYZANTINE}/{N_PARTIES} corrupted parties)")
print("=" * 60)
byzantine_strategies = {
    "FedAvg (mean)": MeanAggregation(),
    "Trimmed mean (10 %)": TrimmedMeanAggregation(trim_fraction=0.1),
    "Coordinate median": MedianAggregation(),
}
byzantine_histories: dict[str, list[float]] = {}
for name, strat in byzantine_strategies.items():
    losses = run_scenario(strat, byzantine=True, label=name)
    byzantine_histories[name] = losses
    print(f"  {name:<28}  final MSE = {losses[-1]:.5f}")

# ---------------------------------------------------------------------------
# Visualise — one chart per scenario
# ---------------------------------------------------------------------------
viz = TrainingHistoryVisualizer()

viz.plot(
    data=honest_histories,
    title="Aggregation Strategies — Honest Setting",
    xlabel="Round",
    ylabel="Global Test MSE",
)

viz.plot(
    data=byzantine_histories,
    title=f"Aggregation Strategies — {N_BYZANTINE}/{N_PARTIES} Byzantine Parties",
    xlabel="Round",
    ylabel="Global Test MSE",
)
