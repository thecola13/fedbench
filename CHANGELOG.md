# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Byzantine-robust aggregation strategies: `KrumAggregation` (incl. Multi-Krum),
  `BulyanAggregation`, `GeometricMedianAggregation` (RFA), `MedianOfMeansAggregation`,
  `CenteredClippingAggregation`.
- Research-grounded models: `SVMModel` (Pegasos), `LassoRegressionModel`,
  `ElasticNetRegressionModel`, `PoissonRegressionModel`, `HuberRegressionModel`.
- Privacy schemes: `LaplaceDPEncryption` (ε-DP), `PairwiseMaskingEncryption`
  (Bonawitz secure aggregation); finished `AdditiveSecretSharing` with real
  mask cancellation.
- `PrivacyUtilityVisualizer` for DP privacy/utility trade-off curves.
- `is_linear_only` contract: order/distance-based aggregators reject
  additive-masking encryption schemes instead of silently miscomputing.
- Runnable, tested example scripts for every new component.
- Packaging metadata (classifiers, keywords, URLs, `py.typed`), `CITATION.cff`,
  GitHub Actions CI, and README badges.

### Changed
- `Model.evaluate` is now a concrete MSE default on the ABC (classifiers/GLMs override).
- `tqdm` moved from core dependencies to the `[examples]` extra.

## [0.1.0]

### Added
- Initial release: strategy-pattern federated learning simulator (`Environment`,
  `Orchestrator`, `Party`) with linear/ridge/closed-form/logistic/MLP models,
  FedAvg/median/trimmed-mean aggregation, NoEncryption/Gaussian-DP/secret-sharing
  schemes, data loading, and visualizers.
