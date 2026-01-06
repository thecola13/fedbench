import numpy as np
from typing import Tuple, List

def generate_linear_data(
    n_samples: int, 
    n_features: int, 
    noise: float = 0.1, 
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic data for linear regression.
    y = Xw + b + noise
    """
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    
    y = np.dot(X, true_weights) + true_bias + np.random.normal(0, noise, size=n_samples)
    
    return X, y

def split_data(
    X: np.ndarray, 
    y: np.ndarray, 
    n_parties: int, 
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Splits data uniformly among parties.
    """
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    X_splits = np.array_split(X_shuffled, n_parties)
    y_splits = np.array_split(y_shuffled, n_parties)
    
    return list(zip(X_splits, y_splits))
