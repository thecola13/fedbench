"""MNIST federated learning — comprehensive benchmark.

Trains a federated MLP classifier on MNIST across up to 10 parties,
sweeping both aggregation strategies and communication protocols.

Configurations
--------------
  A  FedAvg      + NoEncryption   + IID data          (baseline)
  B  TrimmedMean + NoEncryption   + IID data
  C  Median      + NoEncryption   + IID data
  D  FedAvg      + GaussianDP σ=0.005 + IID data
  E  FedAvg      + SecretSharing  + IID data
  F  FedAvg      + NoEncryption   + Non-IID data       (heterogeneous)

Data distributions
------------------
  IID     — 30 000 training samples shuffled uniformly across 10 parties.
  Non-IID — each party holds ≥80 % samples from 2 "primary" digit classes;
             the remaining 20 % are drawn uniformly from all other classes.

Outputs
-------
  • Figure 1 : per-round global test-accuracy curves
  • Figure 2 : final test-accuracy bar chart
  • Figure 3 : per-party local accuracy (IID vs Non-IID, final round)
  • Console  : formatted summary table

Requirements
------------
    uv sync --extra examples   # installs scikit-learn (for MNIST loading)
    uv run python examples/example_mnist_federated.py
"""

from __future__ import annotations

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Fed-env imports
# ---------------------------------------------------------------------------
from fed_playground import (
    AdditiveSecretSharing,
    GaussianDPEncryption,
    MeanAggregation,
    MedianAggregation,
    MLPClassifierModel,
    NoEncryption,
    Orchestrator,
    Party,
    TrimmedMeanAggregation,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_PARTIES = 10
ROUNDS = 15
INPUT_DIM = 784
HIDDEN_DIM = 64
N_CLASSES = 10
LEARNING_RATE = 0.01
EPOCHS_PER_ROUND = 3
BATCH_SIZE = 64
N_TRAIN = 30_000     # subset of MNIST training set (speed vs accuracy trade-off)
N_TEST = 5_000       # subset of MNIST test set
SEED = 42


# ===========================================================================
# Data helpers
# ===========================================================================

def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST via scikit-learn, normalise to [0, 1], return train/test split.

    Returns:
        Tuple (X_train, y_train, X_test, y_test) where X has shape
        (n, 784) as float32 and y has shape (n,) as int.
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for MNIST loading.\n"
            "Run:  uv sync --extra examples"
        ) from exc

    print("Loading MNIST (this may take a moment on first run)…")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0   # (70000, 784)
    y = mnist.target.astype(int)                  # (70000,)

    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(60_000, size=N_TRAIN, replace=False)
    test_idx = rng.choice(np.arange(60_000, 70_000), size=N_TEST, replace=False)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def iid_split(
    X: np.ndarray,
    y: np.ndarray,
    n_parties: int,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffle and split uniformly (IID)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X_s, y_s = X[perm], y[perm]
    return [
        (X_s[i::n_parties], y_s[i::n_parties])
        for i in range(n_parties)
    ]


def non_iid_split(
    X: np.ndarray,
    y: np.ndarray,
    n_parties: int,
    primary_per_party: int = 2,
    primary_fraction: float = 0.8,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Non-IID split: each party holds *primary_fraction* from its primary classes.

    Primary class assignment (round-robin):
      party 0 → classes 0, 1
      party 1 → classes 2, 3
      …
      party 4 → classes 8, 9
      party 5 → classes 0, 2  (wraps)
      …
    """
    classes = np.arange(N_CLASSES)
    primary = [
        [classes[(i * primary_per_party + j) % N_CLASSES] for j in range(primary_per_party)]
        for i in range(n_parties)
    ]

    rng = np.random.default_rng(seed)
    n_per_party = len(y) // n_parties
    n_primary = int(n_per_party * primary_fraction)
    n_other = n_per_party - n_primary

    # Pre-index samples by class for fast lookup
    class_idx: dict[int, np.ndarray] = {
        c: np.where(y == c)[0] for c in classes
    }
    # Shuffle within each class
    for c in classes:
        rng.shuffle(class_idx[c])
    # Pointers into each class's index list
    class_ptr = {c: 0 for c in classes}

    def take(c: int, k: int) -> np.ndarray:
        """Draw k indices from class c (with recycling)."""
        idx = class_idx[c]
        ptr = class_ptr[c]
        needed = k
        drawn: list[np.ndarray] = []
        while needed > 0:
            available = len(idx) - ptr
            chunk = min(available, needed)
            drawn.append(idx[ptr : ptr + chunk])
            ptr += chunk
            needed -= chunk
            if ptr >= len(idx):
                rng.shuffle(idx)
                ptr = 0
        class_ptr[c] = ptr
        return np.concatenate(drawn)

    splits = []
    for i in range(n_parties):
        # Primary-class samples
        per_class = n_primary // primary_per_party
        primary_indices = np.concatenate([take(c, per_class) for c in primary[i]])

        # IID samples from non-primary classes
        other_classes = [c for c in classes if c not in primary[i]]
        per_other = max(1, n_other // len(other_classes))
        other_indices = np.concatenate([take(c, per_other) for c in other_classes])

        all_idx = np.concatenate([primary_indices, other_indices])
        rng.shuffle(all_idx)
        splits.append((X[all_idx], y[all_idx]))

    return splits


# ===========================================================================
# Federated training
# ===========================================================================

def make_parties(
    data_splits: list[tuple[np.ndarray, np.ndarray]],
    encryption_scheme,
) -> list[Party]:
    """Instantiate one Party per data split, each with a fresh model."""
    parties = []
    for i, (Xi, yi) in enumerate(data_splits):
        model = MLPClassifierModel(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            n_classes=N_CLASSES,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS_PER_ROUND,
            batch_size=BATCH_SIZE,
        )
        parties.append(
            Party(party_id=i, model=model, data=(Xi, yi), encryption_scheme=encryption_scheme)
        )
    return parties


def run_experiment(
    label: str,
    data_splits: list[tuple[np.ndarray, np.ndarray]],
    aggregation_strategy,
    encryption_scheme,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = ROUNDS,
) -> dict:
    """Run one federated experiment and return metrics history.

    Returns:
        dict with keys:
          "global_acc"  — list of global test accuracy per round
          "party_acc"   — list of mean local train accuracy per round
          "party_final" — list of per-party local accuracy after last round
          "runtime_s"   — total wall-clock seconds
    """
    parties = make_parties(data_splits, encryption_scheme)

    # Global eval model (shares the same structure; params set after each round)
    eval_model = MLPClassifierModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_classes=N_CLASSES,
    )

    orch = Orchestrator(
        aggregation_strategy=aggregation_strategy,
        encryption_scheme=encryption_scheme,
    )
    for p in parties:
        orch.register_party(p)

    global_acc: list[float] = []
    party_acc: list[float] = []

    t0 = time.perf_counter()
    for r in range(rounds):
        orch.distribute_model()
        for p in parties:
            p.train_local_model()
        orch.aggregate_models()

        # Evaluate global model on held-out test set
        eval_model.set_parameters(orch.global_model_params)
        g_acc = eval_model.evaluate(X_test, y_test)
        global_acc.append(g_acc)

        # Mean local accuracy (on each party's own training data)
        p_accs = [p.evaluate() for p in parties]
        party_acc.append(float(np.mean(p_accs)))

        print(
            f"  [{label}] round {r + 1:2d}/{rounds}"
            f"  global={g_acc:.3f}  local={party_acc[-1]:.3f}"
        )

    runtime = time.perf_counter() - t0

    # Per-party final accuracy (local data)
    party_final = [p.evaluate() for p in parties]

    return {
        "global_acc": global_acc,
        "party_acc": party_acc,
        "party_final": party_final,
        "runtime_s": runtime,
    }


# ===========================================================================
# Visualisation
# ===========================================================================

COLORS = [
    "#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#00BCD4",
]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def plot_training_curves(
    results: dict[str, dict],
    rounds: int,
) -> None:
    """Figure 1: per-round global test accuracy for each configuration."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, rounds + 1)
    for (label, res), color, marker in zip(results.items(), COLORS, MARKERS):
        ax.plot(
            x, res["global_acc"],
            label=label, color=color, marker=marker,
            markevery=max(1, rounds // 8), linewidth=2, markersize=5,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_title(f"Federated MNIST — Training Curves ({N_PARTIES} parties)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(1, rounds)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_final_accuracy_bar(results: dict[str, dict]) -> None:
    """Figure 2: final test accuracy comparison bar chart."""
    labels = list(results.keys())
    final_global = [res["global_acc"][-1] for res in results.values()]
    final_party = [res["party_acc"][-1] for res in results.values()]
    runtimes = [res["runtime_s"] for res in results.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_g = ax.bar(x - width / 2, final_global, width, label="Global test acc",
                    color=COLORS[0], alpha=0.85)
    bars_p = ax.bar(x + width / 2, final_party, width, label="Avg local train acc",
                    color=COLORS[1], alpha=0.85)

    for bar in bars_g:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )
    for bar in bars_p:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\n({rt:.0f}s)" for lbl, rt in zip(labels, runtimes)],
        fontsize=8,
    )
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Round Accuracy Comparison")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_per_party_accuracy(
    iid_result: dict,
    noniid_result: dict,
    n_parties: int,
) -> None:
    """Figure 3: per-party local accuracy breakdown (IID vs Non-IID)."""
    x = np.arange(n_parties)
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(
        x - width / 2, iid_result["party_final"], width,
        label="IID (config A)", color=COLORS[0], alpha=0.85,
    )
    ax.bar(
        x + width / 2, noniid_result["party_final"], width,
        label="Non-IID (config F)", color=COLORS[4], alpha=0.85,
    )
    ax.axhline(
        np.mean(iid_result["party_final"]), color=COLORS[0],
        linestyle="--", linewidth=1.2, label="IID mean",
    )
    ax.axhline(
        np.mean(noniid_result["party_final"]), color=COLORS[4],
        linestyle="--", linewidth=1.2, label="Non-IID mean",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Party {i}" for i in range(n_parties)], fontsize=8)
    ax.set_ylabel("Local Accuracy")
    ax.set_title("Per-Party Local Accuracy: IID vs Non-IID (final round)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_summary_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = 34
    print()
    print("=" * 80)
    print("FEDERATED MNIST — RESULTS SUMMARY")
    print("=" * 80)
    header = (
        f"{'Configuration':<{col_w}}"
        f"{'Final Global Acc':>18}"
        f"{'Avg Local Acc':>16}"
        f"{'Runtime (s)':>14}"
    )
    print(header)
    print("-" * 80)
    for label, res in results.items():
        print(
            f"{label:<{col_w}}"
            f"{res['global_acc'][-1]:>18.4f}"
            f"{res['party_acc'][-1]:>16.4f}"
            f"{res['runtime_s']:>14.1f}"
        )
    print("=" * 80)

    # Degradation analysis
    baseline = results[next(iter(results))]["global_acc"][-1]
    print("\nDegradation relative to baseline (config A):")
    for label, res in list(results.items())[1:]:
        delta = res["global_acc"][-1] - baseline
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<{col_w - 2}}  {sign}{delta:+.4f}")
    print()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    print(
        f"Dataset: {len(y_train)} train / {len(y_test)} test samples "
        f"| {N_PARTIES} parties | {ROUNDS} rounds\n"
    )

    # Data splits ----------------------------------------------------------------
    iid_splits = iid_split(X_train, y_train, N_PARTIES)
    noniid_splits = non_iid_split(X_train, y_train, N_PARTIES)

    # Print class distribution summary
    print("IID class distribution per party (first 3 parties):")
    for i in range(3):
        counts = np.bincount(iid_splits[i][1], minlength=N_CLASSES)
        print(f"  Party {i}: {counts.tolist()}")
    print()
    print("Non-IID class distribution per party (first 3 parties):")
    for i in range(3):
        counts = np.bincount(noniid_splits[i][1], minlength=N_CLASSES)
        print(f"  Party {i}: {counts.tolist()}")
    print()

    # Experiment configurations --------------------------------------------------
    configs = [
        (
            "A: FedAvg + None (IID)",
            iid_splits,
            MeanAggregation(),
            NoEncryption(),
        ),
        (
            "B: TrimmedMean + None (Non-IID)",
            noniid_splits,
            TrimmedMeanAggregation(trim_fraction=0.1),
            NoEncryption(),
        ),
        (
            "C: Median + None (Non-IID)",
            noniid_splits,
            MedianAggregation(),
            NoEncryption(),
        ),
        (
            "D: FedAvg + GaussianDP (Non-IID)",
            noniid_splits,
            MeanAggregation(),
            GaussianDPEncryption(sigma=0.005, seed=SEED),
        ),
        (
            "E: FedAvg + SecretSharing (Non-IID)",
            noniid_splits,
            MeanAggregation(),
            AdditiveSecretSharing(n_shares=3, seed=SEED),
        ),
        (
            "F: FedAvg + None (Non-IID)",
            noniid_splits,
            MeanAggregation(),
            NoEncryption(),
        )
    ]

    # Run all experiments --------------------------------------------------------
    results: dict[str, dict] = {}
    for label, splits, agg, enc in configs:
        print(f"\n{'─' * 60}")
        print(f"Running: {label}")
        print(f"{'─' * 60}")
        results[label] = run_experiment(
            label=label,
            data_splits=splits,
            aggregation_strategy=agg,
            encryption_scheme=enc,
            X_test=X_test,
            y_test=y_test,
            rounds=ROUNDS,
        )

    # Summary table --------------------------------------------------------------
    print_summary_table(results)

    # Figures --------------------------------------------------------------------
    plot_training_curves(results, ROUNDS)
    plot_final_accuracy_bar(results)
    plot_per_party_accuracy(
        iid_result=results["A: FedAvg + None (IID)"],
        noniid_result=results["F: FedAvg + None (Non-IID)"],
        n_parties=N_PARTIES,
    )


if __name__ == "__main__":
    main()
