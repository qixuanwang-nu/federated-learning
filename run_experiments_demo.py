"""
Demo version of experiments with fewer rounds for faster execution
This demonstrates the full workflow with reduced training time
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


def main_demo():
    """Run demo experiments with fewer rounds"""
    
    print("\n" + "="*80)
    print(" " * 20 + "FEDERATED LEARNING DEMO - PART 1")
    print(" " * 15 + "FedAvg on Federated EMNIST Dataset")
    print("="*80)
    
    os.makedirs('results', exist_ok=True)
    
    # 1. Data Analysis
    print("\n" + "="*80)
    print("STEP 1: DATA DISTRIBUTION ANALYSIS")
    print("="*80 + "\n")
    analyze_data_distribution('Assignment3-data/train_data.npy', output_dir='results')
    
    # 2. Baseline Training (C=10%, E=5) - Reduced rounds for demo
    print("\n" + "="*80)
    print("STEP 2: BASELINE TRAINING (C=10%, E=5, 30 rounds)")
    print("="*80 + "\n")
    
    clients, client_weights, test_images, test_labels = load_federated_data(
        'Assignment3-data/train_data.npy',
        'Assignment3-data/test_data.npy',
        train_ratio=0.8
    )
    
    server = FederatedServer(clients, client_weights)
    clients_per_round = max(1, int(0.1 * len(clients)))
    
    history = server.train(
        num_rounds=30,
        clients_per_round=clients_per_round,
        local_epochs=5,
        batch_size=32,
        lr=0.01,
        seed=42
    )
    
    test_loss, test_acc = server.evaluate(test_images, test_labels)
    
    print(f"\nBaseline Results:")
    print(f"  Train Acc: {history['train_acc'][-1]:.4f}")
    print(f"  Val Acc: {history['val_acc'][-1]:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    
    baseline_results = {
        'C': 0.10,
        'E': 5,
        'history': history,
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }
    
    plot_training_curves(history, output_dir='results', experiment_name='demo_baseline')
    server.save_model('results/demo_baseline_model.pth')
    
    ray.shutdown()
    
    # 3. Hyperparameter Comparison - Reduced configs and rounds
    print("\n" + "="*80)
    print("STEP 3: HYPERPARAMETER COMPARISON (3 configs, 25 rounds each)")
    print("="*80 + "\n")
    
    configs = [
        {'C': 0.05, 'E': 3},
        {'C': 0.10, 'E': 5},
        {'C': 0.10, 'E': 10},
    ]
    
    results_list = [baseline_results]  # Include baseline
    
    for idx, config in enumerate(configs[:-1] if configs[-1]['C'] == baseline_results['C'] and configs[-1]['E'] == baseline_results['E'] else configs):
        if config['C'] == 0.10 and config['E'] == 5:
            continue  # Skip baseline as we already have it
            
        print(f"\nConfig {idx+1}: C={config['C']*100:.0f}%, E={config['E']}")
        
        if ray.is_initialized():
            ray.shutdown()
        
        clients, client_weights, test_images, test_labels = load_federated_data(
            'Assignment3-data/train_data.npy',
            'Assignment3-data/test_data.npy',
            train_ratio=0.8
        )
        
        server = FederatedServer(clients, client_weights)
        clients_per_round = max(1, int(config['C'] * len(clients)))
        
        history = server.train(
            num_rounds=25,
            clients_per_round=clients_per_round,
            local_epochs=config['E'],
            batch_size=32,
            lr=0.01,
            seed=42
        )
        
        test_loss, test_acc = server.evaluate(test_images, test_labels)
        
        print(f"  Results: Train={history['train_acc'][-1]:.4f}, Val={history['val_acc'][-1]:.4f}, Test={test_acc:.4f}")
        
        result = {
            'C': config['C'],
            'E': config['E'],
            'history': history,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        }
        results_list.append(result)
        
        plot_training_curves(history, output_dir='results', experiment_name=f"demo_C{int(config['C']*100)}_E{config['E']}")
        
        ray.shutdown()
    
    compare_hyperparameters(results_list, output_dir='results')
    
    # 4. Final Model with Best Hyperparameters
    print("\n" + "="*80)
    print("STEP 4: FINAL MODEL (C=10%, E=10, 40 rounds)")
    print("="*80 + "\n")
    
    clients, client_weights, test_images, test_labels = load_federated_data(
        'Assignment3-data/train_data.npy',
        'Assignment3-data/test_data.npy',
        train_ratio=0.8
    )
    
    server = FederatedServer(clients, client_weights)
    clients_per_round = max(1, int(0.1 * len(clients)))
    
    history = server.train(
        num_rounds=40,
        clients_per_round=clients_per_round,
        local_epochs=10,
        batch_size=32,
        lr=0.01,
        seed=42
    )
    
    test_loss, test_acc = server.evaluate(test_images, test_labels)
    
    print(f"\n{'='*80}")
    print("FINAL MODEL RESULTS")
    print(f"{'='*80}")
    print(f"  Final Training Accuracy: {history['train_acc'][-1]:.4f} ({history['train_acc'][-1]*100:.2f}%)")
    print(f"  Final Validation Accuracy: {history['val_acc'][-1]:.4f} ({history['val_acc'][-1]*100:.2f}%)")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*80}\n")
    
    final_results = {
        'C': 0.10,
        'E': 10,
        'num_rounds': 40,
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }
    
    with open('results/demo_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    plot_training_curves(history, output_dir='results', experiment_name='demo_final')
    server.save_model('results/demo_final_model.pth')
    
    ray.shutdown()
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETED!")
    print("="*80)
    print(f"\nResults saved in: ./results/")
    print(f"  - Data analysis plots")
    print(f"  - Training curves")
    print(f"  - Hyperparameter comparison")
    print(f"  - Model weights")
    print("\nTo run full experiments with more rounds, use: python run_experiments.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main_demo()

