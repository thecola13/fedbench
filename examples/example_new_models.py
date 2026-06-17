"""Federated learning with new model types.

Demonstrates RidgeRegressionModel, LogisticRegressionModel, and
MLPRegressorModel inside the standard Environment / Party workflow.

Usage
-----
    uv run python examples/example_new_models.py
"""

import numpy as np

from fed_playground import (
    ClosedFormLinearRegressionModel,
    Environment,
    LogisticRegressionModel,
    MeanAggregation,
    MLPRegressorModel,
    NoEncryption,
    Orchestrator,
    Party,
    RidgeRegressionModel,
    TrainingHistoryVisualizer,
)

ROUNDS = 15
N_PARTIES = 4
N_SAMPLES = 400
N_FEATURES = 6
SEED = 42


# ---------------------------------------------------------------------------
# 1. Ridge regression (regression task)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Ridge Regression (α=0.5)")
print("=" * 60)

ridge_env = Environment(
    n_parties=N_PARTIES,
    n_features=N_FEATURES,
    n_samples=N_SAMPLES,
    encryption_scheme=NoEncryption(),
    aggregation_strategy=MeanAggregation(),
    model_class=RidgeRegressionModel,
    model_params={"alpha": 0.5},
)
ridge_history = ridge_env.run_simulation(rounds=ROUNDS)

print(f"Initial global loss : {ridge_history['global_loss'][0]:.4f}")
print(f"Final global loss   : {ridge_history['global_loss'][-1]:.4f}")

# ---------------------------------------------------------------------------
# 2. Closed-form linear regression (baseline for comparison)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Closed-form Linear Regression (baseline)")
print("=" * 60)

linear_env = Environment(
    n_parties=N_PARTIES,
    n_features=N_FEATURES,
    n_samples=N_SAMPLES,
    encryption_scheme=NoEncryption(),
    aggregation_strategy=MeanAggregation(),
    model_class=ClosedFormLinearRegressionModel,
)
linear_history = linear_env.run_simulation(rounds=ROUNDS)

print(f"Initial global loss : {linear_history['global_loss'][0]:.4f}")
print(f"Final global loss   : {linear_history['global_loss'][-1]:.4f}")

# ---------------------------------------------------------------------------
# 3. MLP Regressor
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MLP Regressor (hidden_dim=32, epochs=10)")
print("=" * 60)

mlp_env = Environment(
    n_parties=N_PARTIES,
    n_features=N_FEATURES,
    n_samples=N_SAMPLES,
    encryption_scheme=NoEncryption(),
    aggregation_strategy=MeanAggregation(),
    model_class=MLPRegressorModel,
    model_params={"hidden_dim": 32, "learning_rate": 0.005, "epochs": 10},
)
mlp_history = mlp_env.run_simulation(rounds=ROUNDS)

print(f"Initial global loss : {mlp_history['global_loss'][0]:.4f}")
print(f"Final global loss   : {mlp_history['global_loss'][-1]:.4f}")

# ---------------------------------------------------------------------------
# 4. Logistic Regression (binary classification task)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Logistic Regression (binary classification)")
print("=" * 60)

# Generate a separable binary classification dataset.
rng = np.random.default_rng(SEED)
X_cls = rng.normal(size=(N_SAMPLES, N_FEATURES))
true_w = rng.normal(size=N_FEATURES)
y_cls = (X_cls @ true_w > 0).astype(float)

# LogisticRegressionModel.evaluate returns accuracy, not MSE —
# we use the Party / Orchestrator API directly for a cleaner demo.
parties = []
n_per_party = N_SAMPLES // N_PARTIES
for i in range(N_PARTIES):
    start = i * n_per_party
    end = start + n_per_party
    m = LogisticRegressionModel(input_dim=N_FEATURES, learning_rate=0.1, epochs=20)
    p = Party(
        party_id=i,
        model=m,
        data=(X_cls[start:end], y_cls[start:end]),
        encryption_scheme=NoEncryption(),
    )
    parties.append(p)

scheme = NoEncryption()
orch = Orchestrator(
    aggregation_strategy=MeanAggregation(),
    encryption_scheme=scheme,
)
for p in parties:
    orch.register_party(p)

print(f"{'Round':>5}  {'Avg local acc':>14}")
for r in range(ROUNDS):
    orch.distribute_model()
    for p in parties:
        p.train_local_model()
    orch.aggregate_models()
    # Distribute the freshly aggregated params so each party can evaluate them.
    for p in parties:
        p.update_model(orch.global_model_params)
    accs = [p.evaluate() for p in parties]
    print(f"{r + 1:>5}  {np.mean(accs):>14.4f}")

# ---------------------------------------------------------------------------
# 5. Visualise regression training curves
# ---------------------------------------------------------------------------
viz = TrainingHistoryVisualizer()
viz.plot(
    data={
        "Ridge (global)": ridge_history["global_loss"],
        "Linear CF (global)": linear_history["global_loss"],
        "MLP (global)": mlp_history["global_loss"],
    },
    title="Model Comparison — Global Test MSE",
    xlabel="Round",
    ylabel="MSE",
)
