import numpy as np
import os
from fed_playground import Environment, NoEncryption, MeanAggregation, LinearRegressionModel, ClosedFormLinearRegressionModel, DataLoader

def main():
    print("Welcome to the Accredited Federated Learning Playground!")
    
    # Configuration
    n_parties = 3
    rounds = 5
    
    # Check for data
    data_file = 'test_data.csv'
    if not os.path.exists(data_file):
        print(f"File {data_file} not found. Please run 'python create_csv.py' first or I will generate defaults internally.")
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
