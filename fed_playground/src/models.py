import abc
import numpy as np
from typing import Tuple

class Model(abc.ABC):
    """
    Abstract base class for machine learning models.
    """

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data.
        """
        pass

    @abc.abstractmethod
    def get_parameters(self) -> np.ndarray:
        """
        Return the current model parameters as a flattened numpy array.
        """
        pass

    @abc.abstractmethod
    def set_parameters(self, params: np.ndarray) -> None:
        """
        Update the model parameters.
        """
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model and return a metric (e.g., accuracy, MSE).
        """
        pass

class LinearRegressionModel(Model):
    """
    A simple linear regression model using numpy.
    Parameters are [weights, bias].
    """
    def __init__(self, input_dim: int, learning_rate: float = 0.01, epochs: int = 1):
        self.input_dim = input_dim
        self.lr = learning_rate
        self.epochs = epochs
        # Initialize weights and bias
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        if n_samples == 0:
            return

        for _ in range(self.epochs):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute gradients (MSE loss)
            # Loss = (1/n) * sum((y - y_pred)^2)
            # dLoss/dw = -(2/n) * sum(x * (y - y_pred))
            # dLoss/db = -(2/n) * sum(y - y_pred)
            
            error = y - y_pred
            dw = -(2 / n_samples) * np.dot(X.T, error)
            db = -(2 / n_samples) * np.sum(error)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def get_parameters(self) -> np.ndarray:
        # Concatenate weights and bias
        return np.concatenate([self.weights, [self.bias]])

    def set_parameters(self, params: np.ndarray) -> None:
        self.weights = params[:-1]
        self.bias = params[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        # Return Mean Squared Error
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse

class ClosedFormLinearRegressionModel(Model):
    """
    Linear regression model using the Closed Form solution (Normal Equation).
    w = (X^T X)^{-1} X^T y
    """
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        # Weights + Bias
        self.params = np.zeros(input_dim + 1)

    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        # Add a column of ones for the bias term
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))
        return np.hstack([X, ones])

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] == 0:
            return
            
        X_b = self._add_bias_term(X)
        
        # Use pseudo-inverse for stability: theta = pinv(X) * y
        # Or standard: theta = inv(X.T @ X) @ X.T @ y
        # We'll use lstsq which is robust.
        self.params, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)

    def get_parameters(self) -> np.ndarray:
        return self.params

    def set_parameters(self, params: np.ndarray) -> None:
        self.params = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = self._add_bias_term(X)
        return np.dot(X_b, self.params)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse

