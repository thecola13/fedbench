import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class DataLoader:
    """
    Handles loading data from files and preparing it for the environment.
    
    Supports three input modes:
    1. file_path: Load data from CSV file
    2. dataframe: Use an existing pandas DataFrame
    3. X, y: Use numpy arrays or pandas Series/DataFrames directly
    """
    def __init__(
        self, 
        file_path: Optional[str] = None, 
        target_column: str = "target", 
        feature_columns: Optional[List[str]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.dataframe = dataframe
        self.X = X
        self.y = y
        
        # Validate that at least one input method is provided
        input_methods = sum([
            file_path is not None,
            dataframe is not None,
            (X is not None and y is not None)
        ])
        
        if input_methods == 0:
            raise ValueError("Must provide either file_path, dataframe, or (X, y).")
        if input_methods > 1:
            raise ValueError("Provide only one of: file_path, dataframe, or (X, y).")
        
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads data and returns features and target as numpy arrays.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and target (y) as numpy arrays
        """
        # If X and y are provided directly, convert to numpy if needed
        if self.X is not None and self.y is not None:
            X = self.X if isinstance(self.X, np.ndarray) else np.array(self.X)
            y = self.y if isinstance(self.y, np.ndarray) else np.array(self.y)
            return X, y
        
        # Load from dataframe or file
        if self.dataframe is not None:
             df = self.dataframe.copy()
        elif self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path)
        else:
            # Fallback to csv or raise error. 
            # For simplicity, let's treat it as csv.
            df = pd.read_csv(self.file_path)
            
        # Select Features
        if self.feature_columns:
            X = df[self.feature_columns].values
        else:
            # Use all columns except target
            cols = [c for c in df.columns if c != self.target_column]
            X = df[cols].values
            
        # Select Target
        y = df[self.target_column].values
        
        return X, y
