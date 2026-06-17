# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/). Python must be ≥3.11,<3.13 (the `fhe` extra pins this; `concrete-ml` has no 3.13 wheels).

```bash
uv sync                      # core deps
uv sync --extra dev          # + pytest, black, isort, ruff, mypy
uv sync --extra fhe          # + concrete-ml (FHE; 3.11/3.12 only)
uv sync --extra examples     # + scikit-learn (used by some example scripts)

uv run pytest                            # full suite (addopts = -v)
uv run pytest tests/test_models.py       # one file
uv run pytest tests/test_models.py::test_name   # one test
uv run pytest -k "aggregation"           # by keyword

uv run ruff check .          # lint
uv run black . && uv run isort .         # format
uv run mypy fed_playground               # type-check

uv run python examples/basic_simulation.py --parties 4 --rounds 10
```

Set `FED_LOG_LEVEL=INFO` (or `DEBUG`) to see round-by-round progress; the package logs through the standard `logging` module under the `fed_playground` logger.

## Architecture

A pure-NumPy federated learning simulator built around the **strategy pattern**: the four concerns — model, aggregation, encryption, data loading — are independent ABCs you swap without touching the rest. Everything public is re-exported from `fed_playground/__init__.py`; import from there, not from `fed_playground.src.*`.

**Control flow per round** (`Environment` → `Orchestrator` → `Party`):
1. `Orchestrator.distribute_model` broadcasts global params to every registered `Party`.
2. Each `Party.train_local_model` trains its local `Model` on its data shard, then encrypts its params.
3. `Orchestrator.aggregate_models` collects encrypted updates and combines them via the `AggregationStrategy`, producing the new global params.

`Environment` is the high-level driver most users touch — it generates/splits data, instantiates one `model_class` per party plus a global model, wires up the orchestrator, and exposes `run_simulation`/`run_round`. Training history lands in `self.history`.

**The three swappable ABCs** (each in `fed_playground/src/`), plus a `Visualizer` ABC in `visualization.py`:
- `Model` (`models.py`) — `train(X, y)` / `predict(X)`; `evaluate` defaults to MSE on the ABC, classifiers/GLMs override it (accuracy, Poisson deviance). Implementations: linear (GD/closed-form/ridge), logistic, MLP regressor/classifier, plus research models `SVMModel` (Pegasos), `LassoRegressionModel`, `ElasticNetRegressionModel`, `PoissonRegressionModel`. All carry params as a single NumPy array so aggregation is model-agnostic.
- `AggregationStrategy` (`aggregation.py`) — `aggregate(...)`. `MeanAggregation` (FedAvg) is the default; Byzantine-robust options: `MedianAggregation`, `TrimmedMeanAggregation`, `KrumAggregation`, `BulyanAggregation`, `GeometricMedianAggregation`, `MedianOfMeansAggregation`. Shared helpers `_require_plaintext_updates` / `_stack_plaintext` / `_krum_scores` live at module top.
- `EncryptionScheme` (`encryption.py`) — `encrypt` / `decrypt` / `aggregate`. The scheme's `aggregate` runs over *encrypted* params, so the orchestrator and every party must share the same scheme instance type. `NoEncryption` (baseline), `GaussianDPEncryption`, `LaplaceDPEncryption`, `AdditiveSecretSharing`, `PairwiseMaskingEncryption`.

Because encryption defines its own `aggregate`, encrypted aggregation and the `AggregationStrategy` are two distinct layers — don't conflate them.

**`is_linear_only` flag** (class attr on `EncryptionScheme`, default `False`): masking schemes (`AdditiveSecretSharing`, `PairwiseMaskingEncryption`) set it `True` because individual shares are meaningless — only their sum reconstructs. The order/distance-based aggregators check it and raise rather than silently computing garbage over masked shares. Only `MeanAggregation` is sound over masked updates.

New research classes cite their source in the docstring; see the README "Research-grounded components" table. Each has a runnable `examples/example_*.py` demo (examples are gitignored but present on disk).

## Conventions

- `X`, `y`, `X_train` etc. uppercase ML names are deliberate; ruff's `N803`/`N806` are disabled for them. Full type annotations are expected on library code (ruff `ANN`), but not on `tests/` or `examples/`.
- `tests/` mirrors the `src/` layout one-to-one (`test_<module>.py`).
- `examples/` prefixed `thesis_*` / using `THESIS_DATA_DIR` are private research scripts that read an external data dir; they are not needed for the published demos.
