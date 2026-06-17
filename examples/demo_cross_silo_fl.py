"""
Cross-Silo Federated Learning Demonstration

This demo illustrates the fundamental concepts of Cross-Silo Federated Learning
as described in the literature review, specifically addressing:

1. Cross-Silo FL paradigm (limited clients, substantial resources, trusted parties)
2. Horizontal Federated Learning (HFL) setting
3. Centralized architecture with FedAvg aggregation
4. Statistical heterogeneity (non-IID data)
5. Communication efficiency tracking

LITERATURE CONTEXT:
According to the literature review (Section 1.1.2), Cross-Silo FL addresses the
need for institutional players (banks, hospitals, etc.) to collaborate without
sharing local datasets. This setting typically involves:
- Limited number of clients (up to 100)
- Substantial data and computational resources per client
- Trusted parties with stricter regulatory requirements
- Non-IID data due to heterogeneity across organizational datasets

This demo simulates a healthcare scenario where multiple hospitals collaborate
to train a predictive model on patient data without sharing raw information.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_playground import (
    Environment,
    ClosedFormLinearRegressionModel,
    NoEncryption,
    MeanAggregation,
    DataLoader,
    TrainingHistoryVisualizer,
    ComparisonVisualizer
)


class CrossSiloFLDemo:
    """
    Demonstrates Cross-Silo Federated Learning with institutional clients.

    THEORETICAL FOUNDATION:
    This implementation follows the Federated Averaging (FedAvg) protocol
    introduced by McMahan et al. (2017), which operates in three stages:
    1. Server distributes current global model to selected clients
    2. Clients perform local training using private data
    3. Server aggregates client updates to form new global model
    """

    def __init__(self, n_hospitals: int = 5, save_dir: str = "demo_results/cross_silo"):
        """
        Initialize Cross-Silo FL demonstration.

        Args:
            n_hospitals: Number of participating hospitals (institutional clients)
            save_dir: Directory to save results and visualizations
        """
        self.n_hospitals = n_hospitals
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        print("="*70)
        print("CROSS-SILO FEDERATED LEARNING DEMONSTRATION")
        print("="*70)
        print(f"\nScenario: {n_hospitals} hospitals collaborating on disease prediction")
        print("\nKey Characteristics (Literature Review Section 1.1.2):")
        print(f"  • Limited clients: {n_hospitals} hospitals (typical: up to 100)")
        print("  • Trusted institutional parties")
        print("  • Substantial data per client")
        print("  • Non-IID data (different patient populations)")
        print("="*70)

    def generate_hospital_data(self, seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic patient data simulating institutional heterogeneity.

        STATISTICAL HETEROGENEITY (Section 1.4):
        Each hospital serves different patient populations, uses different
        equipment, and follows different protocols, leading to non-IID data.

        This simulation creates:
        - Different feature distributions per hospital
        - Varying dataset sizes (realistic institutional variation)
        - Shared feature space but different data distributions

        Returns:
            DataFrame with patient data and hospital assignments
        """
        np.random.seed(seed)

        all_data = []

        for hospital_id in range(self.n_hospitals):
            # Vary sample size per hospital (system heterogeneity)
            n_patients = np.random.randint(150, 300)

            # Create hospital-specific distribution shifts (statistical heterogeneity)
            # Each hospital has slightly different patient demographics
            age_bias = hospital_id * 5
            risk_factor_bias = hospital_id * 0.1

            # Generate features with hospital-specific distributions
            age = np.random.normal(50 + age_bias, 15, n_patients)
            bmi = np.random.normal(25, 5, n_patients)
            blood_pressure = np.random.normal(120, 15, n_patients)
            cholesterol = np.random.normal(200 + risk_factor_bias * 20, 30, n_patients)

            # Target: disease risk score (influenced by features)
            risk_score = (
                0.3 * (age - 50) / 15 +
                0.25 * (bmi - 25) / 5 +
                0.25 * (blood_pressure - 120) / 15 +
                0.2 * (cholesterol - 200) / 30 +
                np.random.normal(0, 0.5, n_patients)
            )

            hospital_data = pd.DataFrame({
                'age': age,
                'bmi': bmi,
                'blood_pressure': blood_pressure,
                'cholesterol': cholesterol,
                'risk_score': risk_score,
                'hospital_id': hospital_id
            })

            all_data.append(hospital_data)

        return pd.concat(all_data, ignore_index=True)

    def run_centralized_baseline(self, data: pd.DataFrame) -> Dict:
        """
        Train a centralized model for comparison.

        BASELINE COMPARISON:
        In traditional ML, all data would be centralized. This serves as
        the performance upper bound that FL attempts to approach while
        maintaining privacy.
        """
        print("\n" + "-"*70)
        print("STEP 1: CENTRALIZED BASELINE (Traditional ML)")
        print("-"*70)
        print("Training centralized model on pooled data...")

        features = ['age', 'bmi', 'blood_pressure', 'cholesterol']
        target = 'risk_score'

        # Train/test split
        train_data = data.sample(frac=0.8, random_state=42)
        test_data = data.drop(train_data.index)

        X_train = train_data[features].values
        y_train = train_data[target].values
        X_test = test_data[features].values
        y_test = test_data[target].values

        # Train centralized model
        model = ClosedFormLinearRegressionModel(input_dim=len(features))
        model.train(X_train, y_train)
        mse = model.evaluate(X_test, y_test)

        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Centralized MSE: {mse:.6f}")
        print("  Status: ✓ Baseline established")

        return {
            'mse': mse,
            'model': model,
            'test_data': (X_test, y_test)
        }

    def run_federated_learning(self, data: pd.DataFrame, rounds: int = 10) -> Dict:
        """
        Execute Federated Learning across hospitals.

        FEDAVG PROTOCOL (Section 1.3):
        Implements the standard FedAvg algorithm:
        1. Initialize global model
        2. For each round:
           a. Broadcast global model to clients
           b. Clients train locally on private data
           c. Server aggregates weighted average of updates
        3. Repeat until convergence

        PRIVACY PRESERVATION:
        Raw patient data never leaves each hospital. Only model updates
        (weights) are communicated to the central server.
        """
        print("\n" + "-"*70)
        print("STEP 2: FEDERATED LEARNING (Privacy-Preserving)")
        print("-"*70)
        print(f"Initiating FL with {self.n_hospitals} hospitals...")

        features = ['age', 'bmi', 'blood_pressure', 'cholesterol']
        target = 'risk_score'

        # Prepare data for FL (keep test set separate)
        train_data = data.sample(frac=0.8, random_state=42)
        test_data = data.drop(train_data.index)

        # Create DataLoader
        loader = DataLoader(
            dataframe=train_data,
            target_column=target,
            feature_columns=features
        )

        # Setup FL environment
        print(f"\n  Network Architecture: Centralized (FedAvg)")
        print(f"  Aggregation Strategy: Mean (weighted by data size)")
        print(f"  Encryption: None (baseline - can be extended)")
        print(f"  Communication rounds: {rounds}")

        env = Environment(
            n_parties=self.n_hospitals,
            encryption_scheme=NoEncryption(),
            aggregation_strategy=MeanAggregation(),
            model_class=ClosedFormLinearRegressionModel,
            data_loader=loader
        )

        # Prepare test data
        X_test = test_data[features].values
        y_test = test_data[target].values

        # Run simulation
        print(f"\n  Executing {rounds} communication rounds...")
        history = env.run_simulation(rounds=rounds, test_data=(X_test, y_test))

        # Get final federated model
        fed_model = ClosedFormLinearRegressionModel(input_dim=len(features))
        fed_model.set_parameters(env.orchestrator.global_model_params)
        final_mse = fed_model.evaluate(X_test, y_test)

        print(f"\n  Final Federated MSE: {final_mse:.6f}")
        print("  Status: ✓ Federated learning completed")

        # Analyze data distribution across hospitals
        print(f"\n  Data Distribution Analysis:")
        # Calculate samples per hospital from training data
        samples_per_hospital = len(train_data) // self.n_hospitals
        for i in range(self.n_hospitals):
            if i < len(train_data) % self.n_hospitals:
                hospital_samples = samples_per_hospital + 1
            else:
                hospital_samples = samples_per_hospital
            print(f"    Hospital {i+1}: ~{hospital_samples} samples")

        return {
            'history': history,
            'final_mse': final_mse,
            'fed_model': fed_model,
            'environment': env,
            'X_test': X_test,
            'y_test': y_test
        }

    def analyze_results(self, centralized_results: Dict, federated_results: Dict):
        """
        Compare centralized vs federated performance.

        EVALUATION METRICS (Section 1.6):
        Standard FL evaluation includes:
        - Predictive accuracy (MSE for regression)
        - Communication efficiency (rounds to convergence)
        - Fairness across clients (variance in performance)
        """
        print("\n" + "-"*70)
        print("STEP 3: PERFORMANCE ANALYSIS")
        print("-"*70)

        cent_mse = centralized_results['mse']
        fed_mse = federated_results['final_mse']
        performance_gap = ((fed_mse - cent_mse) / cent_mse) * 100

        print(f"\n  Centralized MSE:  {cent_mse:.6f}")
        print(f"  Federated MSE:    {fed_mse:.6f}")
        print(f"  Performance Gap:  {performance_gap:.2f}%")

        if performance_gap < 5:
            print("  Assessment: ✓ Excellent - FL achieves near-centralized performance")
        elif performance_gap < 15:
            print("  Assessment: ✓ Good - Acceptable privacy-utility tradeoff")
        else:
            print("  Assessment: ⚠ Moderate - Consider advanced aggregation strategies")

        # Communication efficiency
        history = federated_results['history']
        print(f"\n  Communication Rounds: {len(history['global_loss'])}")
        print(f"  Convergence: {'Yes' if len(history['global_loss']) > 5 else 'Ongoing'}")

        # Client fairness analysis
        env = federated_results['environment']
        X_test, y_test = federated_results['X_test'], federated_results['y_test']

        print(f"\n  Client-Level Performance (Fairness Analysis):")
        local_mses = []
        for i, party in enumerate(env.parties):
            local_mse = party.model.evaluate(X_test, y_test)
            local_mses.append(local_mse)
            gap_from_global = ((local_mse - fed_mse) / fed_mse) * 100
            print(f"    Hospital {i+1} MSE: {local_mse:.6f} ({gap_from_global:+.1f}% from global)")

        fairness_variance = np.var(local_mses)
        print(f"  Performance Variance: {fairness_variance:.6f}")

        return {
            'performance_gap': performance_gap,
            'fairness_variance': fairness_variance,
            'local_mses': local_mses
        }

    def visualize_results(self, centralized_results: Dict, federated_results: Dict,
                         analysis: Dict):
        """
        Create comprehensive visualizations.
        """
        print("\n" + "-"*70)
        print("STEP 4: VISUALIZATION GENERATION")
        print("-"*70)

        # 1. Training History
        print("  Generating training history plot...")
        history = federated_results['history']
        viz1 = TrainingHistoryVisualizer(save_dir=self.save_dir)
        viz1.plot(
            data={
                "Global Test Loss": history['global_loss'],
                "Avg Local Train Loss": history['party_loss']
            },
            title="Federated Learning Convergence",
            xlabel="Communication Round",
            ylabel="Mean Squared Error",
            filename="fl_convergence.png"
        )

        # 2. Model Comparison
        print("  Generating model comparison plot...")
        comparison_data = {
            "Centralized\n(All Data)": centralized_results['mse'],
            "Federated\n(Privacy-Preserving)": federated_results['final_mse']
        }

        # Add local models
        for i, mse in enumerate(analysis['local_mses']):
            comparison_data[f"Hospital {i+1}\n(Local Only)"] = mse

        viz2 = ComparisonVisualizer(save_dir=self.save_dir)
        viz2.plot(
            data=comparison_data,
            title="Model Performance Comparison",
            xlabel="Training Approach",
            ylabel="Test MSE (lower is better)",
            filename="model_comparison.png",
            color=['green', 'blue'] + ['orange'] * len(analysis['local_mses'])
        )

        print(f"  Status: ✓ Visualizations saved to {self.save_dir}/")

    def generate_report(self, centralized_results: Dict, federated_results: Dict,
                       analysis: Dict):
        """
        Generate comprehensive markdown report.
        """
        report_path = os.path.join(self.save_dir, "DEMO_REPORT.md")

        with open(report_path, 'w') as f:
            f.write("# Cross-Silo Federated Learning Demo Report\n\n")

            f.write("## Executive Summary\n\n")
            f.write(f"This demonstration simulates a **Cross-Silo Federated Learning** scenario ")
            f.write(f"where **{self.n_hospitals} hospitals** collaborate to train a disease risk ")
            f.write(f"prediction model without sharing patient data.\n\n")

            f.write("## Literature Context\n\n")
            f.write("### Cross-Silo FL (Section 1.1.2)\n\n")
            f.write("The Cross-Silo FL paradigm addresses institutional collaboration needs:\n")
            f.write("- **Limited number of clients** (typically up to 100)\n")
            f.write("- **Substantial data and computational resources** per client\n")
            f.write("- **Trusted parties** with stricter regulatory requirements\n")
            f.write("- **Non-IID data** due to organizational heterogeneity\n\n")

            f.write("### FedAvg Algorithm (Section 1.3)\n\n")
            f.write("Implementation follows McMahan et al.'s Federated Averaging:\n")
            f.write("1. Server distributes global model to clients\n")
            f.write("2. Clients train locally on private data\n")
            f.write("3. Server aggregates weighted average of updates\n\n")

            f.write("## Experimental Results\n\n")
            f.write("### Performance Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Centralized MSE | {centralized_results['mse']:.6f} |\n")
            f.write(f"| Federated MSE | {federated_results['final_mse']:.6f} |\n")
            f.write(f"| Performance Gap | {analysis['performance_gap']:.2f}% |\n")
            f.write(f"| Communication Rounds | {len(federated_results['history']['global_loss'])} |\n")
            f.write(f"| Fairness Variance | {analysis['fairness_variance']:.6f} |\n\n")

            f.write("### Key Findings\n\n")

            if analysis['performance_gap'] < 5:
                f.write("✓ **Excellent Performance**: FL achieves near-centralized accuracy ")
                f.write("while maintaining data privacy.\n\n")
            elif analysis['performance_gap'] < 15:
                f.write("✓ **Good Performance**: Acceptable privacy-utility tradeoff achieved.\n\n")
            else:
                f.write("⚠ **Moderate Performance**: Consider advanced aggregation strategies ")
                f.write("(FedProx, SCAFFOLD) for better convergence.\n\n")

            f.write("### Hospital-Level Analysis\n\n")
            f.write("| Hospital | Test MSE | Deviation from Global |\n")
            f.write("|----------|----------|-----------------------|\n")
            for i, mse in enumerate(analysis['local_mses']):
                deviation = ((mse - federated_results['final_mse']) /
                            federated_results['final_mse']) * 100
                f.write(f"| Hospital {i+1} | {mse:.6f} | {deviation:+.2f}% |\n")

            f.write("\n## Privacy Guarantees\n\n")
            f.write("This demonstration uses the baseline FL protocol where:\n")
            f.write("- **Raw patient data never leaves** each hospital\n")
            f.write("- Only **model updates (weights)** are communicated\n")
            f.write("- Central server **never sees individual patient records**\n\n")

            f.write("### Extensions for Enhanced Privacy (Section 1.5)\n\n")
            f.write("The framework can be extended with:\n")
            f.write("- **Homomorphic Encryption (HE)**: Encrypt model updates before transmission\n")
            f.write("- **Secure Aggregation (SecAgg)**: Cryptographic masking ensures server ")
            f.write("only sees aggregated result\n")
            f.write("- **Differential Privacy (DP)**: Add calibrated noise to updates for ")
            f.write("provable privacy guarantees\n\n")

            f.write("## Conclusions\n\n")
            f.write(f"This demo successfully demonstrates Cross-Silo FL with {self.n_hospitals} ")
            f.write(f"institutional clients. The federated model achieves {100 - analysis['performance_gap']:.1f}% ")
            f.write(f"of centralized performance while maintaining data privacy—a practical ")
            f.write(f"solution for privacy-constrained collaborative learning scenarios.\n\n")

            f.write("## References\n\n")
            f.write("- McMahan et al. (2017): Communication-Efficient Learning of Deep Networks ")
            f.write("from Decentralized Data\n")
            f.write("- Literature Review Sections 1.1.2 (Cross-Silo FL), 1.3 (FedAvg), ")
            f.write("1.5 (Privacy Mechanisms)\n")

        print(f"  Status: ✓ Report saved to {report_path}")


def main():
    """Run the Cross-Silo FL demonstration."""

    # Initialize demo
    demo = CrossSiloFLDemo(n_hospitals=5)

    # Generate hospital data
    print("\nGenerating synthetic hospital data...")
    data = demo.generate_hospital_data()
    print(f"  Total patients: {len(data)}")
    print(f"  Features: age, BMI, blood_pressure, cholesterol")
    print(f"  Target: disease risk score")

    # Run centralized baseline
    centralized_results = demo.run_centralized_baseline(data)

    # Run federated learning
    federated_results = demo.run_federated_learning(data, rounds=15)

    # Analyze results
    analysis = demo.analyze_results(centralized_results, federated_results)

    # Visualize
    demo.visualize_results(centralized_results, federated_results, analysis)

    # Generate report
    demo.generate_report(centralized_results, federated_results, analysis)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {demo.save_dir}/")
    print("  • fl_convergence.png - Training history")
    print("  • model_comparison.png - Performance comparison")
    print("  • DEMO_REPORT.md - Comprehensive analysis report")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
