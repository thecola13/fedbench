# Examples

This directory contains example scripts demonstrating various features of the federated learning playground.

## Available Examples

### 1. [`basic_simulation.py`](file:///Users/cola/Desktop/Projects/fed_env/examples/basic_simulation.py)

**Basic federated learning simulation**

Demonstrates:
- Setting up a federated learning environment
- Loading data from CSV or using synthetic data
- Running a multi-party simulation
- Tracking training progress

**Usage:**
```bash
python examples/basic_simulation.py
```

### 2. [`divergence_analysis.py`](file:///Users/cola/Desktop/Projects/fed_env/examples/divergence_analysis.py)

**Analyze how federated learning diverges from centralized training**

Demonstrates:
- Comparing federated vs centralized models
- Three analysis modes: instances diff, data diff, fixed data
- Using DivergenceVisualizer for comprehensive plots
- Metrics: norm difference, MSE difference, MSE ratio

**Usage:**

**Mode 1 - Vary number of parties:**
```bash
python examples/divergence_analysis.py \
    --data-path examples/test_data.csv \
    --features feature_0 feature_1 \
    --target target \
    --instances-diff \
    --rounds 5
```

**Mode 2 - Vary data per party:**
```bash
python examples/divergence_analysis.py \
    --data-path examples/test_data.csv \
    --features feature_0 feature_1 \
    --target target \
    --data-diff \
    --rounds 5
```

**Mode 3 - Fixed total data:**
```bash
python examples/divergence_analysis.py \
    --data-path examples/test_data.csv \
    --features feature_0 feature_1 \
    --target target \
    --fixed-data \
    --total-data-points 500 \
    --rounds 5
```

### 3. [`visualization_demo.py`](file:///Users/cola/Desktop/Projects/fed_env/examples/visualization_demo.py)

**Comprehensive visualization system demonstration**

Demonstrates all three visualizers:
- TrainingHistoryVisualizer
- ComparisonVisualizer
- DivergenceVisualizer

**Usage:**
```bash
python examples/visualization_demo.py
```

Output saved to `demo_results/` directory.

## Test Data

### [`test_data.csv`](file:///Users/cola/Desktop/Projects/fed_env/examples/test_data.csv)

Sample dataset with 200 rows and 5 features for testing examples.

**To generate new test data:**
```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 200
n_features = 5

X = np.random.randn(n_samples, n_features)
true_weights = np.random.randn(n_features)
y = np.dot(X, true_weights) + 0.5 + np.random.normal(0, 0.1, size=n_samples)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['target'] = y
df.to_csv('examples/test_data.csv', index=False)
```
