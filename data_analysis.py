"""
Data Analysis and Visualization for Federated EMNIST Dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os


def analyze_data_distribution(train_path: str, output_dir: str = 'results'):
    """
    Analyze and visualize the data distribution across classes and clients.
    
    Args:
        train_path: Path to training data file
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_data = np.load(train_path, allow_pickle=True)
    
    print("="*60)
    print("FEDERATED EMNIST DATA ANALYSIS")
    print("="*60)
    
    # Overall statistics
    total_samples = sum([len(train_data[i]['labels']) for i in range(len(train_data))])
    print(f"\nOverall Statistics:")
    print(f"  Total clients: {len(train_data)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Average samples per client: {total_samples / len(train_data):.1f}")
    
    # Collect all labels
    all_labels = []
    for i in range(len(train_data)):
        labels = np.array(train_data[i]['labels'])
        all_labels.extend(labels.tolist())
    
    all_labels = np.array(all_labels)
    
    # Overall class distribution
    print(f"\n{'='*60}")
    print("OVERALL CLASS DISTRIBUTION")
    print(f"{'='*60}")
    
    unique_classes = np.unique(all_labels)
    print(f"  Number of classes: {len(unique_classes)}")
    print(f"  Class range: [{unique_classes.min()}, {unique_classes.max()}]")
    
    class_counts = Counter(all_labels)
    print(f"\n  Class distribution:")
    print(f"    Min samples per class: {min(class_counts.values())}")
    print(f"    Max samples per class: {max(class_counts.values())}")
    print(f"    Average samples per class: {np.mean(list(class_counts.values())):.1f}")
    print(f"    Median samples per class: {np.median(list(class_counts.values())):.1f}")
    
    # Plot overall class distribution
    fig, ax = plt.subplots(figsize=(14, 6))
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    ax.bar(classes, counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Class Label', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Overall Class Distribution Across All Clients', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_class_distribution.png'), dpi=300)
    print(f"\n  Saved plot: {output_dir}/overall_class_distribution.png")
    plt.close()
    
    # Analyze 5 selected clients
    print(f"\n{'='*60}")
    print("INDIVIDUAL CLIENT ANALYSIS (5 Selected Clients)")
    print(f"{'='*60}")
    
    # Select 5 clients with varying data sizes
    client_sizes = [(i, len(train_data[i]['labels'])) for i in range(len(train_data))]
    client_sizes.sort(key=lambda x: x[1])
    
    # Select clients: smallest, 25th percentile, median, 75th percentile, largest
    indices = [
        0,  # smallest
        len(client_sizes) // 4,  # 25th percentile
        len(client_sizes) // 2,  # median
        3 * len(client_sizes) // 4,  # 75th percentile
        len(client_sizes) - 1  # largest
    ]
    
    selected_clients = [client_sizes[i][0] for i in indices]
    
    # Create subplot for 5 clients
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, client_id in enumerate(selected_clients):
        labels = np.array(train_data[client_id]['labels'])
        client_class_counts = Counter(labels)
        
        print(f"\nClient {client_id}:")
        print(f"  Total samples: {len(labels)}")
        print(f"  Number of classes: {len(client_class_counts)}")
        print(f"  Class range: [{min(labels)}, {max(labels)}]")
        
        # Top 5 most common classes
        most_common = client_class_counts.most_common(5)
        print(f"  Top 5 classes:")
        for class_id, count in most_common:
            print(f"    Class {class_id}: {count} samples ({100*count/len(labels):.1f}%)")
        
        # Plot for this client
        ax = axes[idx]
        classes = sorted(client_class_counts.keys())
        counts = [client_class_counts[c] for c in classes]
        
        ax.bar(classes, counts, color='coral', alpha=0.7)
        ax.set_xlabel('Class Label', fontsize=10)
        ax.set_ylabel('Number of Samples', fontsize=10)
        ax.set_title(f'Client {client_id} (N={len(labels)} samples, {len(client_class_counts)} classes)', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Remove the 6th subplot (we only have 5 clients)
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'client_class_distributions.png'), dpi=300)
    print(f"\n  Saved plot: {output_dir}/client_class_distributions.png")
    plt.close()
    
    # Data heterogeneity analysis
    print(f"\n{'='*60}")
    print("DATA HETEROGENEITY ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate number of classes per client
    classes_per_client = []
    samples_per_client = []
    
    for i in range(len(train_data)):
        labels = np.array(train_data[i]['labels'])
        classes_per_client.append(len(np.unique(labels)))
        samples_per_client.append(len(labels))
    
    print(f"\n  Samples per client:")
    print(f"    Min: {min(samples_per_client)}")
    print(f"    Max: {max(samples_per_client)}")
    print(f"    Mean: {np.mean(samples_per_client):.1f}")
    print(f"    Std: {np.std(samples_per_client):.1f}")
    
    print(f"\n  Classes per client:")
    print(f"    Min: {min(classes_per_client)}")
    print(f"    Max: {max(classes_per_client)}")
    print(f"    Mean: {np.mean(classes_per_client):.1f}")
    print(f"    Std: {np.std(classes_per_client):.1f}")
    
    # Plot distribution of samples and classes per client
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(samples_per_client, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Samples', fontsize=12)
    ax1.set_ylabel('Number of Clients', fontsize=12)
    ax1.set_title('Distribution of Samples per Client', fontsize=13, fontweight='bold')
    ax1.axvline(np.mean(samples_per_client), color='red', linestyle='--', 
               label=f'Mean: {np.mean(samples_per_client):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(classes_per_client, bins=range(min(classes_per_client), max(classes_per_client) + 2), 
            color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Classes', fontsize=12)
    ax2.set_ylabel('Number of Clients', fontsize=12)
    ax2.set_title('Distribution of Classes per Client', fontsize=13, fontweight='bold')
    ax2.axvline(np.mean(classes_per_client), color='red', linestyle='--', 
               label=f'Mean: {np.mean(classes_per_client):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_heterogeneity.png'), dpi=300)
    print(f"\n  Saved plot: {output_dir}/data_heterogeneity.png")
    plt.close()
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


def plot_training_curves(history: dict, output_dir: str = 'results', 
                        experiment_name: str = 'fedavg'):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
        experiment_name: Name for the experiment (used in filename)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    rounds = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(rounds, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(rounds, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(rounds, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(rounds, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_training_curves.png'), dpi=300)
    print(f"Saved training curves: {output_dir}/{experiment_name}_training_curves.png")
    plt.close()


def compare_hyperparameters(results: list, output_dir: str = 'results'):
    """
    Compare training results for different hyperparameter settings.
    
    Args:
        results: List of dictionaries containing experiment results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training accuracy comparison
    ax = axes[0, 0]
    for result in results:
        label = f"C={result['C']}, E={result['E']}"
        ax.plot(result['history']['train_acc'], label=label, linewidth=2)
    ax.set_xlabel('Communication Round', fontsize=11)
    ax.set_ylabel('Training Accuracy', fontsize=11)
    ax.set_title('Training Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy comparison
    ax = axes[0, 1]
    for result in results:
        label = f"C={result['C']}, E={result['E']}"
        ax.plot(result['history']['val_acc'], label=label, linewidth=2)
    ax.set_xlabel('Communication Round', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final accuracies bar chart
    ax = axes[1, 0]
    labels = [f"C={r['C']}, E={r['E']}" for r in results]
    train_accs = [r['history']['train_acc'][-1] for r in results]
    val_accs = [r['history']['val_acc'][-1] for r in results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, train_accs, width, label='Training', alpha=0.8)
    ax.bar(x + width/2, val_accs, width, label='Validation', alpha=0.8)
    ax.set_xlabel('Hyperparameter Setting', fontsize=11)
    ax.set_ylabel('Final Accuracy', fontsize=11)
    ax.set_title('Final Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Training efficiency (rounds to reach 80% val accuracy)
    ax = axes[1, 1]
    rounds_to_80 = []
    for result in results:
        val_acc = result['history']['val_acc']
        rounds = next((i+1 for i, acc in enumerate(val_acc) if acc >= 0.80), len(val_acc))
        rounds_to_80.append(rounds)
    
    ax.bar(labels, rounds_to_80, alpha=0.8, color='steelblue')
    ax.set_xlabel('Hyperparameter Setting', fontsize=11)
    ax.set_ylabel('Rounds to 80% Val Accuracy', fontsize=11)
    ax.set_title('Training Efficiency Comparison', fontsize=13, fontweight='bold')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_comparison.png'), dpi=300)
    print(f"Saved hyperparameter comparison: {output_dir}/hyperparameter_comparison.png")
    plt.close()


if __name__ == '__main__':
    # Run data analysis
    analyze_data_distribution('Assignment3-data/train_data.npy')

