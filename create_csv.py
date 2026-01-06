import numpy as np
import pandas as pd

# Generate data
np.random.seed(42)
n_samples = 200
n_features = 5
X = np.random.randn(n_samples, n_features)
true_weights = np.random.randn(n_features)
y = np.dot(X, true_weights) + 0.5 + np.random.normal(0, 0.1, size=n_samples)

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['target'] = y

filename = 'test_data.csv'
df.to_csv(filename, index=False)
print(f"Created {filename} with shape {df.shape}")
