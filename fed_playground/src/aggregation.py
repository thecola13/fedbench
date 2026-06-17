"""Aggregation strategies for the federated learning orchestrator.

Each strategy receives the list of (possibly encrypted) model updates collected
from all parties and returns a single aggregated result.  The result is stored
as the new global model in the :class:`~fed_playground.src.orchestrator.Orchestrator`.
"""

import abc
from typing import Any

import numpy as np

from .encryption import EncryptionScheme


def _require_plaintext_updates(
    strategy_name: str, encryption_scheme: EncryptionScheme
) -> None:
    """Reject masking schemes whose per-party shares carry no usable value.

    Robust aggregators (median, trimmed mean, Krum, geometric median) inspect
    individual party updates — impossible once additive masking has hidden them
    (only their sum reconstructs).  Such schemes support linear aggregation only.
    """
    if encryption_scheme.is_linear_only:
        raise ValueError(
            f"{strategy_name} needs per-party plaintext values, but "
            f"{type(encryption_scheme).__name__} masks them — order/distance "
            "statistics cannot be computed over additive shares. Use "
            "MeanAggregation."
        )


def _stack_plaintext(
    encrypted_models: list[Any], encryption_scheme: EncryptionScheme
) -> np.ndarray:
    """Decrypt any non-plaintext entries and stack into ``(n_parties, n_params)``."""
    arrays = [
        encryption_scheme.decrypt(p) if not isinstance(p, np.ndarray) else p
        for p in encrypted_models
    ]
    return np.stack(arrays, axis=0)


def _krum_scores(stacked: np.ndarray, n_byzantine: int) -> np.ndarray:
    """Krum score per update: sum of the closest ``n - f - 2`` squared distances.

    Lower is better — a low score means the update sits in a dense cluster and
    is therefore unlikely to be an adversarial outlier.
    """
    n = stacked.shape[0]
    diff = stacked[:, None, :] - stacked[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)  # (n, n)
    n_neighbours = max(1, n - n_byzantine - 2)
    scores = np.empty(n)
    for i in range(n):
        others = np.delete(sq_dist[i], i)
        scores[i] = np.sort(others)[:n_neighbours].sum()
    return scores


class AggregationStrategy(abc.ABC):
    """Abstract base class for model aggregation strategies."""

    @abc.abstractmethod
    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Aggregate a list of party model updates into a single result.

        Args:
            encrypted_models: Non-empty list of (possibly encrypted) parameter
                vectors, one per registered party.
            encryption_scheme: The scheme in use; may be needed to perform
                homomorphic operations on ciphertexts.

        Returns:
            Aggregated model parameters in the same representation as the inputs
            (plaintext numpy array for :class:`~fed_playground.src.encryption.NoEncryption`,
            ciphertext otherwise).

        Raises:
            ValueError: If *encrypted_models* is empty.
        """


class TrimmedMeanAggregation(AggregationStrategy):
    """Byzantine-robust aggregation via coordinate-wise trimmed mean.

    For each parameter dimension, removes the ``trim_fraction`` lowest and
    highest values before computing the mean.  This provides robustness
    against a bounded fraction of malicious or corrupted party updates.

    Requires plaintext (numpy) parameters; FHE ciphertexts are not supported.

    Args:
        trim_fraction: Fraction of parties to trim from each tail (default ``0.1``).
            Must satisfy ``2 * trim_fraction < 1``.
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5).")
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Compute the coordinate-wise trimmed mean of party updates.

        Args:
            encrypted_models: Non-empty list of numpy parameter vectors.
            encryption_scheme: Active encryption scheme (used for summation
                to obtain plaintext arrays before trimming).

        Returns:
            Trimmed-mean parameter vector as a numpy array.

        Raises:
            ValueError: If *encrypted_models* is empty or parameters are not
                numpy arrays.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")
        _require_plaintext_updates("TrimmedMeanAggregation", encryption_scheme)

        stacked = _stack_plaintext(encrypted_models, encryption_scheme)
        n = stacked.shape[0]
        k = int(np.floor(n * self.trim_fraction))

        if k == 0:
            return stacked.mean(axis=0)

        # Sort along the party axis and slice out the k lowest and k highest.
        stacked_sorted = np.sort(stacked, axis=0)
        trimmed = stacked_sorted[k : n - k]
        return trimmed.mean(axis=0)


class MedianAggregation(AggregationStrategy):
    """Byzantine-robust aggregation via coordinate-wise median.

    For each parameter dimension the median value across all party updates
    is selected.  This is tolerant of up to ``(n_parties - 1) / 2``
    arbitrarily corrupted parties.

    Requires plaintext (numpy) parameters; FHE ciphertexts are not supported.
    """

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Compute the coordinate-wise median of party updates.

        Args:
            encrypted_models: Non-empty list of numpy parameter vectors.
            encryption_scheme: Active encryption scheme.

        Returns:
            Median parameter vector as a numpy array.

        Raises:
            ValueError: If *encrypted_models* is empty.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")
        _require_plaintext_updates("MedianAggregation", encryption_scheme)

        return np.median(_stack_plaintext(encrypted_models, encryption_scheme), axis=0)


class MeanAggregation(AggregationStrategy):
    """Federated averaging (FedAvg) — compute the mean of all party updates.

    When :class:`~fed_playground.src.encryption.NoEncryption` is used the
    result is the true arithmetic mean.

    For schemes that support only homomorphic *addition* (not scalar division),
    the aggregated result is the **sum** and the caller is responsible for
    dividing by the number of parties after decryption.  This is documented
    behaviour and a deliberate trade-off for FHE compatibility.
    """

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Compute the mean (or sum for opaque ciphertexts) of party updates.

        Args:
            encrypted_models: Non-empty list of party parameter vectors.
            encryption_scheme: Active encryption scheme.

        Returns:
            Arithmetic mean if parameters are numpy arrays; element-wise sum
            otherwise (for FHE ciphertexts that do not support scalar division).

        Raises:
            ValueError: If *encrypted_models* is empty.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")

        summed = encryption_scheme.aggregate(encrypted_models)

        # Divide only when we have a plaintext numpy array.
        # FHE ciphertexts cannot be divided here without the private key;
        # the receiving party must divide by N after decryption.
        if isinstance(summed, np.ndarray):
            return summed / len(encrypted_models)

        return summed


class KrumAggregation(AggregationStrategy):
    """Byzantine-robust aggregation via the Krum / Multi-Krum rule.

    Reference: Blanchard, El Mhamdi, Guerraoui & Stainer, "Machine Learning with
    Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017.

    For ``n`` party updates with at most ``n_byzantine`` (f) adversaries, Krum
    scores each update by the sum of squared Euclidean distances to its
    ``n - f - 2`` nearest neighbours, then selects the single update with the
    smallest score — the one most "supported" by a majority cluster, which by
    construction cannot be a far-flung Byzantine outlier.  **Multi-Krum**
    (``n_selected > 1``) averages the ``n_selected`` lowest-scoring updates,
    trading a little robustness for lower variance.

    Krum's guarantee holds when ``2f + 2 < n``.  With too few parties for that
    bound this falls back to using all other updates as neighbours (still a
    sensible "most central update" selector).

    Requires plaintext (numpy) updates; incompatible with additive-masking
    schemes (``is_linear_only``).

    Args:
        n_byzantine: Assumed number of Byzantine parties ``f`` (default ``1``).
        n_selected: Updates to average (1 = classic Krum; >1 = Multi-Krum).
    """

    def __init__(self, n_byzantine: int = 1, n_selected: int = 1) -> None:
        if n_byzantine < 0:
            raise ValueError("n_byzantine must be >= 0.")
        if n_selected < 1:
            raise ValueError("n_selected must be >= 1.")
        self.n_byzantine = n_byzantine
        self.n_selected = n_selected

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Select (Multi-)Krum update(s) and return their mean.

        Args:
            encrypted_models: Non-empty list of numpy parameter vectors.
            encryption_scheme: Active encryption scheme.

        Returns:
            The chosen Krum update, or the mean of the top ``n_selected``.

        Raises:
            ValueError: If *encrypted_models* is empty or the scheme masks updates.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")
        _require_plaintext_updates("KrumAggregation", encryption_scheme)

        stacked = _stack_plaintext(encrypted_models, encryption_scheme)
        n = stacked.shape[0]
        if n <= 2:
            # Not enough updates to form a neighbourhood; fall back to the mean.
            return stacked.mean(axis=0)

        scores = _krum_scores(stacked, self.n_byzantine)
        n_sel = min(self.n_selected, n)
        chosen = np.argsort(scores)[:n_sel]
        return stacked[chosen].mean(axis=0)


class GeometricMedianAggregation(AggregationStrategy):
    """Robust aggregation via the geometric (spatial) median — the RFA rule.

    Reference: Pillutla, Kakade & Harchaoui, "Robust Aggregation for Federated
    Learning", IEEE Transactions on Signal Processing, 2022.

    Returns the point ``v`` minimising the sum of Euclidean distances to the
    party updates, ``argmin_v Σ_i ‖v − x_i‖``.  Unlike the coordinate-wise
    median this is rotation-equivariant, and it has an asymptotic breakdown
    point of 0.5 (robust until half the mass is adversarial).  Computed with
    the smoothed Weiszfeld iteration

        ``v ← (Σ_i x_i / max(ε, ‖v − x_i‖)) / (Σ_i 1 / max(ε, ‖v − x_i‖))``

    initialised at the mean.  The ``ε`` floor keeps the update well-defined when
    an iterate coincides with a data point.

    Requires plaintext (numpy) updates; incompatible with additive-masking
    schemes (``is_linear_only``).

    Args:
        max_iter: Maximum Weiszfeld iterations (default ``100``).
        eps: Smoothing floor on distances / convergence tolerance (default ``1e-8``).
    """

    def __init__(self, max_iter: int = 100, eps: float = 1e-8) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        self.max_iter = max_iter
        self.eps = eps

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Compute the geometric median of the party updates.

        Args:
            encrypted_models: Non-empty list of numpy parameter vectors.
            encryption_scheme: Active encryption scheme.

        Returns:
            Geometric-median parameter vector as a numpy array.

        Raises:
            ValueError: If *encrypted_models* is empty or the scheme masks updates.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")
        _require_plaintext_updates("GeometricMedianAggregation", encryption_scheme)

        stacked = _stack_plaintext(encrypted_models, encryption_scheme)
        v = stacked.mean(axis=0)
        for _ in range(self.max_iter):
            dist = np.maximum(self.eps, np.linalg.norm(stacked - v, axis=1))
            weights = 1.0 / dist
            v_new = (weights[:, None] * stacked).sum(axis=0) / weights.sum()
            if np.linalg.norm(v_new - v) <= self.eps:
                v = v_new
                break
            v = v_new
        return v


class BulyanAggregation(AggregationStrategy):
    """Strong Byzantine-robust aggregation via Bulyan (Krum + trimmed mean).

    Reference: El Mhamdi, Guerraoui & Rouault, "The Hidden Vulnerability of
    Distributed Learning in Byzantium", ICML 2018.

    Bulyan closes a loophole in Krum: a single Krum-selected update can still be
    nudged far along one coordinate by a clever attacker.  It runs in two stages:

    1. **Selection** — recursively apply Krum to pick ``m = n - 2f`` candidate
       updates (select the Krum-best, remove it, repeat), discarding the most
       outlying updates.
    2. **Coordinate-wise trimmed mean** — over the selected set, for each
       coordinate keep the ``β = m - 2f`` values closest to that coordinate's
       median and average them, bounding any single dimension's corruption.

    The guarantee requires ``n ≥ 4f + 3``.  With fewer parties this clamps the
    selection/averaging counts to stay well-defined (degrading gracefully toward
    a coordinate-wise median).

    Requires plaintext (numpy) updates; incompatible with additive-masking
    schemes (``is_linear_only``).

    Args:
        n_byzantine: Assumed number of Byzantine parties ``f`` (default ``1``).
    """

    def __init__(self, n_byzantine: int = 1) -> None:
        if n_byzantine < 0:
            raise ValueError("n_byzantine must be >= 0.")
        self.n_byzantine = n_byzantine

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Run Bulyan selection + coordinate-wise trimmed mean.

        Args:
            encrypted_models: Non-empty list of numpy parameter vectors.
            encryption_scheme: Active encryption scheme.

        Returns:
            The Bulyan-aggregated parameter vector.

        Raises:
            ValueError: If *encrypted_models* is empty or the scheme masks updates.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")
        _require_plaintext_updates("BulyanAggregation", encryption_scheme)

        stacked = _stack_plaintext(encrypted_models, encryption_scheme)
        n = stacked.shape[0]
        f = self.n_byzantine
        m = max(1, n - 2 * f)  # selection-set size

        # Stage 1: recursive Krum selection.
        remaining = list(range(n))
        selected: list[int] = []
        while len(selected) < m and remaining:
            sub = stacked[remaining]
            if sub.shape[0] <= 2:
                selected.extend(remaining)
                break
            scores = _krum_scores(sub, f)
            best_local = int(np.argmin(scores))
            selected.append(remaining.pop(best_local))
        sel = stacked[selected]  # (m', n_params)

        # Stage 2: coordinate-wise trimmed mean over the selected set.
        beta = max(1, sel.shape[0] - 2 * f)
        median = np.median(sel, axis=0)
        # Indices of the beta values closest to the median, per coordinate.
        order = np.argsort(np.abs(sel - median), axis=0)[:beta]
        closest = np.take_along_axis(sel, order, axis=0)
        return closest.mean(axis=0)


class MedianOfMeansAggregation(AggregationStrategy):
    """Robust aggregation via the median-of-means estimator.

    References: Nemirovski & Yudin 1983; Lugosi & Mendelson, "Mean estimation
    and regression under heavy-tailed distributions: a survey", 2019.

    Partitions the party updates into ``n_buckets`` groups, averages each group,
    then takes the coordinate-wise **median of the bucket means**.  Averaging
    within buckets reduces variance (sub-Gaussian concentration even for
    heavy-tailed updates); the outer median discards buckets contaminated by
    Byzantine parties.  Robust as long as a majority of buckets are clean — i.e.
    ``n_buckets > 2f`` for ``f`` adversaries.

    Requires plaintext (numpy) updates; incompatible with additive-masking
    schemes (``is_linear_only``).

    Args:
        n_buckets: Number of buckets ``k`` (default ``5``); clamped to the party
            count.  Choose ``k > 2f`` to tolerate ``f`` Byzantine parties.
        seed: RNG seed for the random partition (default ``0``).
    """

    def __init__(self, n_buckets: int = 5, seed: int = 0) -> None:
        if n_buckets < 1:
            raise ValueError("n_buckets must be >= 1.")
        self.n_buckets = n_buckets
        self.seed = seed

    def aggregate(
        self,
        encrypted_models: list[Any],
        encryption_scheme: EncryptionScheme,
    ) -> Any:
        """Average within random buckets, then take the coordinate-wise median.

        Args:
            encrypted_models: Non-empty list of numpy parameter vectors.
            encryption_scheme: Active encryption scheme.

        Returns:
            The median-of-means parameter vector.

        Raises:
            ValueError: If *encrypted_models* is empty or the scheme masks updates.
        """
        if not encrypted_models:
            raise ValueError("encrypted_models must not be empty.")
        _require_plaintext_updates("MedianOfMeansAggregation", encryption_scheme)

        stacked = _stack_plaintext(encrypted_models, encryption_scheme)
        n = stacked.shape[0]
        k = min(self.n_buckets, n)
        rng = np.random.default_rng(self.seed)
        buckets = np.array_split(rng.permutation(n), k)
        bucket_means = np.stack([stacked[idx].mean(axis=0) for idx in buckets], axis=0)
        return np.median(bucket_means, axis=0)
