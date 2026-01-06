import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class DataLoader:
    """
    Handles loading data from files and preparing it for the environment.
    """
    def __init__(
        self, 
        file_path: str, 
        target_column: str, 
        feature_columns: Optional[List[str]] = None
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.feature_columns = feature_columns
        
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads data from CSV, separates features and target.
        Returns (X, y) as numpy arrays.
        """
        # Read the file
        # Check extension or assume CSV
        if self.file_path.endswith('.csv'):
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
