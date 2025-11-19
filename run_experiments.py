"""
Main script to run all experiments for Federated Learning Assignment
"""

import numpy as np
import ray
import json
import os
from fedavg import load_federated_data, FederatedServer
from data_analysis import (
    analyze_data_distribution, 
    plot_training_curves, 
    compare_hyperparameters
)


def experiment_1_data_analysis():
    """
    Experiment 1: Analyze data distribution
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: DATA DISTRIBUTION ANALYSIS")
    print("="*80 + "\n")
    
    analyze_data_distribution('Assignment3-data/train_data.npy', output_dir='results')


def experiment_2_baseline_training():
    """
    Experiment 2: Train with C=10% and plot training curves
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: BASELINE TRAINING (C=10%, E=5)")
    print("="*80 + "\n")
    
    # Load data
    clients, client_weights, test_images, test_labels = load_federated_data(
        'Assignment3-data/train_data.npy',
        'Assignment3-data/test_data.npy',
        train_ratio=0.8
    )
    
    # Create server
    server = FederatedServer(clients, client_weights)
    
    # Training parameters
    num_rounds = 100
    clients_per_round = max(1, int(0.1 * len(clients)))  # C = 10%
    local_epochs = 5
    batch_size = 32
    lr = 0.01
    
    print(f"Configuration:")
    print(f"  C (client fraction): 10% = {clients_per_round}/{len(clients)} clients")
    print(f"  E (local epochs): {local_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Communication rounds: {num_rounds}\n")
    
    # Train
    history = server.train(
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=42
    )
    
    # Evaluate
    test_loss, test_acc = server.evaluate(test_images, test_labels)
    
    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS (C=10%, E=5)")
    print(f"{'='*60}")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'C': 0.10,
        'E': 5,
        'num_rounds': num_rounds,
        'history': history,
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }
    
    with open('results/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, output_dir='results', experiment_name='baseline_C10_E5')
    
    # Save model
    server.save_model('results/baseline_model.pth')
    print(f"\nModel saved to: results/baseline_model.pth")
    
    ray.shutdown()
    
    return results


def experiment_3_hyperparameter_comparison():
    """
    Experiment 3: Compare different values of C and E
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: HYPERPARAMETER COMPARISON")
    print("="*80 + "\n")
    
    # Define hyperparameter configurations to test
    configs = [
        {'C': 0.05, 'E': 3},   # Fewer clients, fewer epochs
        {'C': 0.05, 'E': 5},   # Fewer clients, baseline epochs
        {'C': 0.10, 'E': 3},   # Baseline clients, fewer epochs
        {'C': 0.10, 'E': 5},   # Baseline (from experiment 2)
        {'C': 0.10, 'E': 10},  # Baseline clients, more epochs
    ]
    
    results_list = []
    num_rounds = 80  # Slightly fewer rounds for comparison experiments
    
    for idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {idx+1}/{len(configs)}: C={config['C']*100:.0f}%, E={config['E']}")
        print(f"{'='*60}")
        
        # Load data (reinitialize Ray for each experiment)
        if ray.is_initialized():
            ray.shutdown()
        
        clients, client_weights, test_images, test_labels = load_federated_data(
            'Assignment3-data/train_data.npy',
            'Assignment3-data/test_data.npy',
            train_ratio=0.8
        )
        
        # Create server
        server = FederatedServer(clients, client_weights)
        
        # Calculate number of clients
        clients_per_round = max(1, int(config['C'] * len(clients)))
        
        print(f"\nTraining with:")
        print(f"  C = {config['C']*100:.0f}% ({clients_per_round}/{len(clients)} clients)")
        print(f"  E = {config['E']} epochs")
        print(f"  Rounds = {num_rounds}\n")
        
        # Train
        history = server.train(
            num_rounds=num_rounds,
            clients_per_round=clients_per_round,
            local_epochs=config['E'],
            batch_size=32,
            lr=0.01,
            seed=42
        )
        
        # Evaluate
        test_loss, test_acc = server.evaluate(test_images, test_labels)
        
        print(f"\nResults for C={config['C']}, E={config['E']}:")
        print(f"  Final Training Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"  Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        # Store results
        result = {
            'C': config['C'],
            'E': config['E'],
            'history': history,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        }
        results_list.append(result)
        
        # Plot individual training curve
        experiment_name = f"C{int(config['C']*100)}_E{config['E']}"
        plot_training_curves(history, output_dir='results', experiment_name=experiment_name)
        
        ray.shutdown()
    
    # Compare all hyperparameters
    compare_hyperparameters(results_list, output_dir='results')
    
    # Save comparison results
    # Convert numpy arrays to lists for JSON serialization
    results_for_json = []
    for r in results_list:
        result_json = {
            'C': r['C'],
            'E': r['E'],
            'test_accuracy': r['test_accuracy'],
            'test_loss': r['test_loss'],
            'final_train_acc': r['history']['train_acc'][-1],
            'final_val_acc': r['history']['val_acc'][-1]
        }
        results_for_json.append(result_json)
    
    with open('results/hyperparameter_comparison.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("HYPERPARAMETER COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'C':>6} {'E':>4} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10}")
    print("-" * 80)
    for r in results_list:
        print(f"{r['C']:>6.0%} {r['E']:>4} {r['history']['train_acc'][-1]:>10.4f} "
              f"{r['history']['val_acc'][-1]:>10.4f} {r['test_accuracy']:>10.4f}")
    
    return results_list


def experiment_4_final_evaluation():
    """
    Experiment 4: Train with best hyperparameters and evaluate on test data
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*80 + "\n")
    
    # Based on typical federated learning results, use C=10%, E=10 for final model
    # (more local updates generally help with convergence)
    
    clients, client_weights, test_images, test_labels = load_federated_data(
        'Assignment3-data/train_data.npy',
        'Assignment3-data/test_data.npy',
        train_ratio=0.8
    )
    
    # Create server
    server = FederatedServer(clients, client_weights)
    
    # Training parameters (best configuration)
    num_rounds = 120
    clients_per_round = max(1, int(0.1 * len(clients)))  # C = 10%
    local_epochs = 10  # More epochs for final model
    batch_size = 32
    lr = 0.01
    
    print(f"Final Configuration:")
    print(f"  C (client fraction): 10% = {clients_per_round}/{len(clients)} clients")
    print(f"  E (local epochs): {local_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Communication rounds: {num_rounds}\n")
    
    # Train
    history = server.train(
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=42
    )
    
    # Evaluate on test data
    test_loss, test_acc = server.evaluate(test_images, test_labels)
    
    print(f"\n{'='*80}")
    print(f"FINAL MODEL RESULTS")
    print(f"{'='*80}")
    print(f"Configuration: C=10%, E={local_epochs}, Rounds={num_rounds}")
    print(f"\nTraining Performance:")
    print(f"  Final Training Accuracy: {history['train_acc'][-1]:.4f} ({history['train_acc'][-1]*100:.2f}%)")
    print(f"  Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"\nValidation Performance:")
    print(f"  Final Validation Accuracy: {history['val_acc'][-1]:.4f} ({history['val_acc'][-1]*100:.2f}%)")
    print(f"  Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"\nTest Performance:")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        'C': 0.10,
        'E': local_epochs,
        'num_rounds': num_rounds,
        'batch_size': batch_size,
        'learning_rate': lr,
        'final_train_accuracy': history['train_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'final_val_accuracy': history['val_acc'][-1],
        'final_val_loss': history['val_loss'][-1],
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }
    
    with open('results/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, output_dir='results', experiment_name='final_model')
    
    # Save model
    server.save_model('results/final_model.pth')
    print(f"Final model saved to: results/final_model.pth")
    
    ray.shutdown()
    
    return results


def main():
    """
    Run all experiments in sequence
    """
    print("\n" + "="*80)
    print(" " * 20 + "FEDERATED LEARNING - PART 1")
    print(" " * 15 + "FedAvg on Federated EMNIST Dataset")
    print("="*80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Experiment 1: Data Analysis
    experiment_1_data_analysis()
    
    # Experiment 2: Baseline Training
    baseline_results = experiment_2_baseline_training()
    
    # Experiment 3: Hyperparameter Comparison
    comparison_results = experiment_3_hyperparameter_comparison()
    
    # Experiment 4: Final Evaluation
    final_results = experiment_4_final_evaluation()
    
    # Print overall summary
    print("\n" + "="*80)
    print(" " * 30 + "OVERALL SUMMARY")
    print("="*80)
    print(f"\nBaseline Model (C=10%, E=5):")
    print(f"  Test Accuracy: {baseline_results['test_accuracy']:.4f} ({baseline_results['test_accuracy']*100:.2f}%)")
    
    print(f"\nFinal Model (C=10%, E=10):")
    print(f"  Test Accuracy: {final_results['test_accuracy']:.4f} ({final_results['test_accuracy']*100:.2f}%)")
    
    print(f"\nAll results saved in: ./results/")
    print(f"  - Data analysis plots")
    print(f"  - Training curves for all experiments")
    print(f"  - Hyperparameter comparison plots")
    print(f"  - Trained model weights")
    print(f"  - JSON result files")
    
    print("\n" + "="*80)
    print(" " * 25 + "EXPERIMENTS COMPLETED!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

