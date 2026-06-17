"""Differential-privacy noise and additive secret sharing.

Compares four configurations that differ only in their encryption scheme:

    1. NoEncryption       — plaintext baseline
    2. GaussianDP σ=0.01  — light noise
    3. GaussianDP σ=0.1   — medium noise
    4. GaussianDP σ=0.5   — heavy noise
    5. AdditiveSecretSharing — additive shares (reconstructs exactly)

The experiment illustrates the privacy-utility trade-off: larger σ adds
stronger local differential privacy but increases the global test loss.

Usage
-----
    uv run python examples/example_dp_encryption.py
"""

from fed_playground import (
    AdditiveSecretSharing,
    ClosedFormLinearRegressionModel,
    Environment,
    GaussianDPEncryption,
    MeanAggregation,
    NoEncryption,
    TrainingHistoryVisualizer,
)

ROUNDS = 20
N_PARTIES = 5
N_FEATURES = 8
N_SAMPLES = 500
SEED = 7

configs = {
    "No encryption": NoEncryption(),
    "Gaussian DP σ=0.01": GaussianDPEncryption(sigma=0.01, seed=SEED),
    "Gaussian DP σ=0.1": GaussianDPEncryption(sigma=0.1, seed=SEED),
    "Gaussian DP σ=0.5": GaussianDPEncryption(sigma=0.5, seed=SEED),
    "Additive secret sharing": AdditiveSecretSharing(n_shares=3, seed=SEED),
}

histories: dict[str, list[float]] = {}

for label, scheme in configs.items():
    env = Environment(
        n_parties=N_PARTIES,
        n_features=N_FEATURES,
        n_samples=N_SAMPLES,
        encryption_scheme=scheme,
        aggregation_strategy=MeanAggregation(),
        model_class=ClosedFormLinearRegressionModel,
    )
    hist = env.run_simulation(rounds=ROUNDS)
    histories[label] = hist["global_loss"]
    final = hist["global_loss"][-1]
    print(f"{label:<30}  final MSE = {final:.5f}")

# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------
viz = TrainingHistoryVisualizer()
viz.plot(
    data=histories,
    title="Privacy-Utility Trade-off: Encryption Scheme Comparison",
    xlabel="Round",
    ylabel="Global Test MSE",
)
