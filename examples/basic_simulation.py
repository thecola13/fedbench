"""
Example script demonstrating basic usage of the Federated Learning Playground.

This script shows how to:
- Set up a federated learning environment
- Use either synthetic data or load from CSV
- Run a simulation with multiple parties
- Track global model performance over rounds
"""

import sys
import os
# Add parent directory to path to import fed_playground
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fed_playground import Environment, NoEncryption, MeanAggregation, LinearRegressionModel, ClosedFormLinearRegressionModel, DataLoader

def main():
    print("Welcome to the Accredited Federated Learning Playground!")
    
    # Configuration
    n_parties = 3
    rounds = 5
    
    # Check for data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'test_data.csv')
    
    if not os.path.exists(data_file):
        print(f"File {data_file} not found. Using internal data generation...")
        data_loader = None
        n_features = 5
        n_samples = 100
        print(f"Setting up simulation with internal data generation: {n_parties} parties, {n_features} features.")
    else:
        print(f"Loading data from {data_file}...")
        data_loader = DataLoader(file_path=data_file, target_column='target')
        # n_features and n_samples will be inferred
        n_features = 0 
        n_samples = 0
        print(f"Setting up simulation with external data: {n_parties} parties.")

    
    # Initialize components
    encryption_scheme = NoEncryption()
    aggregation_strategy = MeanAggregation()
    
    # Initialize Environment
    env = Environment(
        n_parties=n_parties,
        n_features=n_features,
        n_samples=n_samples,
        encryption_scheme=encryption_scheme,
        aggregation_strategy=aggregation_strategy,
        model_class=ClosedFormLinearRegressionModel,
        model_params={},
        data_loader=data_loader
    )

    
    # Run Simulation
    print("Starting simulation...")
    env.run_simulation(rounds=rounds)
    
    print("\nSimulation complete.")
    print("History of Global Loss:", env.history['global_loss'])

if __name__ == "__main__":
    main()
