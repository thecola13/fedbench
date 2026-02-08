
import os
import sys
import numpy as np
import pandas as pd
import tqdm
from argparse import ArgumentParser

# Ensure fed_playground imports work
sys.path.append(os.getcwd())

from fed_playground import (
    ClosedFormLinearRegressionModel,
    MeanAggregation,
    DataLoader,
    Environment,
    NoEncryption,
    DivergenceVisualizer
)

def vprint(msg, verbosity, level=1):
    if verbosity >= level:
        print(msg)

def make_supervised_table(df, features, target):
    if isinstance(features, str):
        features = [features]
    
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Features {missing_feats} not found in dataframe.")
    
    if target not in df.columns:
        raise ValueError(f"Target {target} not found in dataframe.")

    X = df[features]
    y = df[target]
    
    return X, y

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, general_model=None):
    """
    Evaluate model and compute divergence metrics if general_model provided.
    """
    X_np = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
    y_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
    
    mse = model.evaluate(X_np, y_np)
    metrics = {"mse": mse}
    
    if general_model:
        # Norm Difference
        w_local = model.get_parameters()
        w_gen = general_model.get_parameters()
        norm_diff = np.linalg.norm(w_local - w_gen)
        metrics["normdiff"] = norm_diff
        
        # MSE Difference and Ratio
        gen_mse = general_model.evaluate(X_np, y_np)
        metrics["msediff"] = mse - gen_mse
        metrics["mseratio"] = mse / gen_mse if gen_mse != 0 else np.inf
        
    return metrics

def do_one_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    num_instances,
    sample_per_instance,
    rounds,
    random_state,
    feature_columns,
    target_column,
    collect_local=True,
):
    end_total = num_instances * sample_per_instance
    if X_train.shape[0] < end_total:
         raise ValueError(f"Not enough training data. Need {end_total}, have {X_train.shape[0]}")

    # 1. Train General Model on all data designated for this experiment
    X_gen = X_train.iloc[:end_total]
    y_gen = y_train.iloc[:end_total]
    
    n_features = X_train.shape[1]
    
    general_model = ClosedFormLinearRegressionModel(input_dim=n_features)
    
    X_gen_np = X_gen.to_numpy()
    y_gen_np = y_gen.to_numpy()
    
    general_model.train(X_gen_np, y_gen_np)
    gen_metrics = evaluate_model(general_model, X_test, y_test)
    general_mse = gen_metrics["mse"]
    print(f"General model MSE: {general_mse:.4f}")

    metrics_per_round = []
    
    for r in tqdm.tqdm(range(rounds), leave=False, desc=f"Rounds (I={num_instances}, D={sample_per_instance})"):
        # Shuffle relevant data
        train_df = pd.concat([X_train, y_train], axis=1)
        # Use a fresh slice of data for the round (shuffled)
        # Note: In standard FL, parties hold static data usually. 
        # But this divergence script simulates re-sampling or shuffling to see robust divergence?
        # The original script shuffled data each round. "Shuffle data for this round".
        # So we emulate that by creating a DataLoader with shuffled data.
        
        train_shuffled = train_df.sample(frac=1, random_state=random_state + r)
        experiment_data = train_shuffled.iloc[:end_total]
        
        # Initialize Environment
        # We pass the subset of data via DataLoader(dataframe=...)
        loader = DataLoader(dataframe=experiment_data, target_column=target_column, feature_columns=feature_columns)
        
        env = Environment(
            n_parties=num_instances,
            encryption_scheme=NoEncryption(),
            aggregation_strategy=MeanAggregation(),
            model_class=ClosedFormLinearRegressionModel,
            data_loader=loader
        )
        
        # Run one round (or multiple? The original script shuffled each round. 
        # If we use Environment, it usually keeps state. 
        # BUT: The original script re-instantiated local models/slices every round loop.
        # "Fit local models... on disjoint slices" inside the loop.
        # So essentially each "round" in the plot script is an independent experiment iteration.
        
        env.setup()
        
        # Run one FL round in the environment (distribute -> train local -> aggregate)
        env.run_round()
        
        # Evaluate
        # We need the aggregated global model from the orchestrator
        agg_params = env.orchestrator.global_model_params
        
        # Create a temp model to evaluate aggregated params
        avg_model = ClosedFormLinearRegressionModel(input_dim=n_features)
        avg_model.set_parameters(agg_params)
        
        round_metrics = evaluate_model(avg_model, X_test, y_test, general_model)
        round_metrics["general_mse"] = general_mse
        
        if collect_local:
            local_mse = []
            local_normdiff = []
            local_msediff = []
            local_mseratio = []
            
            for p in env.parties:
                # p.model is the trained local model
                lm = p.model
                m = evaluate_model(lm, X_test, y_test, general_model)
                local_mse.append(m["mse"])
                local_normdiff.append(m["normdiff"])
                local_msediff.append(m["msediff"])
                local_mseratio.append(m["mseratio"])
                
            round_metrics["local_mse"] = local_mse
            round_metrics["local_normdiff"] = local_normdiff
            round_metrics["local_msediff"] = local_msediff
            round_metrics["local_mseratio"] = local_mseratio
            
        metrics_per_round.append(round_metrics)
        
    return metrics_per_round


def main(args):
    if args.data_path:
        df = pd.read_csv(args.data_path)
    elif os.path.exists("test_data.csv"):
        df = pd.read_csv("test_data.csv")
    else:
        raise FileNotFoundError("Please provide --data-path or ensure test_data.csv exists.")

    X, y = make_supervised_table(df, args.features, args.target)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    vprint(f"Train size: {len(X_train)}, Test size: {len(X_test)}", args.verbosity)

    plotter = DivergenceVisualizer(save_dir=args.save_path)
    
    # Common args for experiment
    exp_args = {
        "X_train": X_train, "y_train": y_train, 
        "X_test": X_test, "y_test": y_test,
        "rounds": args.rounds, "random_state": args.random_state,
        "feature_columns": args.features, "target_column": args.target
    }

    # 1. Instances Diff Mode
    if args.instances_diff:
        num_data_points = args.data_per_instance or (len(X_train) // args.max_instances)
        vprint(f"Mode: Instances Diff. Data per instance: {num_data_points}", args.verbosity)
        
        instance_counts = range(args.min_instances, args.max_instances + 1, args.step_instances)
        
        for num_inst in tqdm.tqdm(instance_counts):
             res = do_one_experiment(num_instances=num_inst, sample_per_instance=num_data_points, **exp_args)
             plotter.add_result(num_inst, res)
             
        if args.save_path:
             save_dir = os.path.join(args.save_path, f"instances_diff_{args.target}")
             plotter.save_dir = save_dir
             plotter.plot("Num Instances", "Number of Instances")
    
    # 2. Data Diff Mode
    elif args.data_diff:
        num_instances = args.instances or (len(X_train) // args.max_data)
        vprint(f"Mode: Data Diff. Num Instances: {num_instances}", args.verbosity)
        
        data_counts = range(args.min_data, args.max_data + 1, args.step_data)
        
        for n_data in tqdm.tqdm(data_counts):
             res = do_one_experiment(num_instances=num_instances, sample_per_instance=n_data, **exp_args)
             plotter.add_result(n_data, res)
             
        if args.save_path:
             save_dir = os.path.join(args.save_path, f"data_diff_{args.target}")
             plotter.save_dir = save_dir
             plotter.plot("Data Per Instance", "Data Amount")

    # 3. Fixed Data Mode
    elif args.fixed_data:
        vprint(f"Mode: Fixed Data. Total Points: {args.total_data_points}", args.verbosity)
        
        instance_counts = range(args.min_instances, args.max_instances + 1, args.step_instances)
        
        for num_inst in tqdm.tqdm(instance_counts):
             samples_per = args.total_data_points // num_inst
             if samples_per < 1: continue
                  
             res = do_one_experiment(num_instances=num_inst, sample_per_instance=samples_per, **exp_args)
             plotter.add_result(num_inst, res)
             
        if args.save_path:
             save_dir = os.path.join(args.save_path, f"fixed_data_{args.target}")
             plotter.save_dir = save_dir
             plotter.plot("Num Instances", "Fixed Data Total")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--features", type=str, nargs="+", required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--save-path", type=str, default="results")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbosity", type=int, default=1)
    
    parser.add_argument("--instances-diff", action="store_true")
    parser.add_argument("--min-instances", type=int, default=2)
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--step-instances", type=int, default=2)
    parser.add_argument("--data-per-instance", type=int, default=None)
    
    parser.add_argument("--data-diff", action="store_true")
    parser.add_argument("--min-data", type=int, default=10)
    parser.add_argument("--max-data", type=int, default=100)
    parser.add_argument("--step-data", type=int, default=10)
    parser.add_argument("--instances", type=int, default=None)
    
    parser.add_argument("--fixed-data", action="store_true")
    parser.add_argument("--total-data-points", type=int, default=1000)
    
    args = parser.parse_args()
    main(args)
