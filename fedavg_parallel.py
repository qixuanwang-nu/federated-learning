"""
Federated Averaging (FedAvg) Implementation with Parallel Ray Actors (Part 2)
This implementation creates Ray Actor instances dynamically for selected clients per round.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import ray
from typing import Dict, List, Tuple
import copy
from tqdm import tqdm


class FederatedNN(nn.Module):
    """
    2-layer fully connected Neural Network with 128 hidden units and ReLU activation.
    Input: 28x28 images (flattened to 784)
    Hidden: 128 units with ReLU
    Output: 62 classes (10 digits + 26 lowercase + 26 uppercase)
    """
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=62):
        super(FederatedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@ray.remote(num_cpus=1, num_gpus=0)  # Each actor uses 1 CPU, 0 GPU (set to 1 if GPU available)
class ParallelFederatedClient:
    """
    Ray Actor for parallel federated client training.
    Each actor handles local training on its private data.
    Created dynamically per round for selected clients only.
    """
    def __init__(self, client_id: int, images: np.ndarray, labels: np.ndarray, 
                 train_ratio: float = 0.8):
        """
        Initialize a federated client actor.
        
        Args:
            client_id: Unique identifier for the client
            images: Local images (numpy array)
            labels: Local labels (numpy array)
            train_ratio: Ratio of data to use for training (rest for validation)
        """
        self.client_id = client_id
        
        # Split data into train and validation
        n_samples = len(images)
        n_train = int(n_samples * train_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Convert to PyTorch tensors
        self.train_images = torch.FloatTensor(images[train_indices])
        self.train_labels = torch.LongTensor(labels[train_indices])
        self.val_images = torch.FloatTensor(images[val_indices])
        self.val_labels = torch.LongTensor(labels[val_indices])
        
        self.n_train = len(self.train_labels)
        self.n_val = len(self.val_labels)
    
    def get_data_size(self) -> Tuple[int, int]:
        """Return the number of training and validation samples."""
        return self.n_train, self.n_val
    
    def local_update(self, global_weights: Dict, epochs: int, 
                    batch_size: int = 32, lr: float = 0.01) -> Tuple[Dict, float, float, float, float]:
        """
        Perform local training on the client's data.
        
        Args:
            global_weights: Global model weights from server
            epochs: Number of local training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            Updated local model weights, train loss, train accuracy, val loss, val accuracy
        """
        # Create model and load global weights
        model = FederatedNN()
        model.load_state_dict(global_weights)
        model.train()
        
        # Setup optimizer and loss function
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Create data loader
        train_dataset = TensorDataset(self.train_images, self.train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Local training
        for epoch in range(epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Calculate training metrics
        train_loss, train_acc = self._evaluate(model, self.train_images, self.train_labels, criterion)
        
        # Calculate validation metrics
        val_loss, val_acc = self._evaluate(model, self.val_images, self.val_labels, criterion)
        
        return model.state_dict(), train_loss, train_acc, val_loss, val_acc
    
    def _evaluate(self, model: nn.Module, images: torch.Tensor, 
                 labels: torch.Tensor, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on given data."""
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels).item()
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == labels).float().mean().item()
        model.train()
        return loss, accuracy


class ParallelFederatedServer:
    """
    Federated learning server with dynamic actor creation for selected clients.
    Creates and destroys actors per round to minimize memory overhead.
    """
    def __init__(self, client_data: List[Tuple[np.ndarray, np.ndarray]], 
                 train_ratio: float = 0.8):
        """
        Initialize the federated server.
        
        Args:
            client_data: List of (images, labels) tuples for all clients
            train_ratio: Train/validation split ratio
        """
        self.client_data = client_data
        self.train_ratio = train_ratio
        self.n_clients = len(client_data)
        
        # Pre-compute client weights (number of training samples)
        print(f"Computing client data sizes...")
        self.client_weights = []
        for i, (images, labels) in enumerate(client_data):
            n_train = int(len(images) * train_ratio)
            self.client_weights.append(n_train)
        
        self.client_weights = np.array(self.client_weights)
        self.total_samples = np.sum(self.client_weights)
        
        print(f"Initialized server with {self.n_clients} clients, "
              f"{self.total_samples} total training samples")
        
        # Initialize global model
        self.global_model = FederatedNN()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def create_actors_for_round(self, selected_indices: List[int]) -> List:
        """
        Create Ray actors for selected clients in this round.
        
        Args:
            selected_indices: Indices of clients selected for this round
            
        Returns:
            List of Ray actor references
        """
        actors = []
        for idx in selected_indices:
            images, labels = self.client_data[idx]
            actor = ParallelFederatedClient.remote(
                client_id=idx,
                images=images,
                labels=labels,
                train_ratio=self.train_ratio
            )
            actors.append(actor)
        return actors
    
    def aggregate_weights(self, local_weights: List[Dict], 
                         selected_indices: List[int]) -> Dict:
        """
        Aggregate local model weights using weighted averaging (FedAvg).
        
        Args:
            local_weights: List of local model weight dictionaries
            selected_indices: Indices of clients that participated in this round
            
        Returns:
            Aggregated global model weights
        """
        # Get weights for selected clients
        selected_weights = self.client_weights[selected_indices]
        total_selected = np.sum(selected_weights)
        
        # Initialize aggregated weights with zeros
        global_weights = copy.deepcopy(local_weights[0])
        for key in global_weights.keys():
            global_weights[key] = torch.zeros_like(global_weights[key])
        
        # Weighted average of local weights
        for local_w, weight in zip(local_weights, selected_weights):
            for key in global_weights.keys():
                global_weights[key] += local_w[key] * (weight / total_selected)
        
        return global_weights
    
    def train(self, num_rounds: int, clients_per_round: int, 
             local_epochs: int, batch_size: int = 32, lr: float = 0.01,
             seed: int = 42) -> Dict:
        """
        Train the global model using FedAvg with dynamic actor creation.
        
        Args:
            num_rounds: Number of communication rounds
            clients_per_round: Number of clients to select per round
            local_epochs: Number of local epochs per client
            batch_size: Batch size for local training
            lr: Learning rate
            seed: Random seed for reproducibility
            
        Returns:
            Training history dictionary
        """
        np.random.seed(seed)
        
        print(f"\n{'='*80}")
        print(f"Starting Federated Training with Parallel Actors")
        print(f"{'='*80}")
        print(f"Number of rounds: {num_rounds}")
        print(f"Clients per round: {clients_per_round}")
        print(f"Local epochs: {local_epochs}")
        print(f"Total clients: {self.n_clients}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Actor creation: Dynamic (per round)")
        print(f"{'='*80}\n")
        
        for round_idx in tqdm(range(num_rounds), desc="Training Rounds"):
            # Select random clients for this round
            selected_indices = np.random.choice(
                self.n_clients, 
                clients_per_round, 
                replace=False
            )
            
            # Create actors for selected clients only
            actors = self.create_actors_for_round(selected_indices)
            
            # Get current global weights
            global_weights = self.global_model.state_dict()
            
            # Parallel local updates using Ray
            futures = [
                actor.local_update.remote(global_weights, local_epochs, batch_size, lr)
                for actor in actors
            ]
            results = ray.get(futures)
            
            # Extract local weights and metrics
            local_weights = [result[0] for result in results]
            train_losses = [result[1] for result in results]
            train_accs = [result[2] for result in results]
            val_losses = [result[3] for result in results]
            val_accs = [result[4] for result in results]
            
            # Clean up actors (important to free memory)
            for actor in actors:
                ray.kill(actor)
            
            # Aggregate weights
            aggregated_weights = self.aggregate_weights(local_weights, selected_indices)
            self.global_model.load_state_dict(aggregated_weights)
            
            # Calculate weighted average metrics
            selected_weights = self.client_weights[selected_indices]
            total_selected = np.sum(selected_weights)
            
            avg_train_loss = np.sum([l * w for l, w in zip(train_losses, selected_weights)]) / total_selected
            avg_train_acc = np.sum([a * w for a, w in zip(train_accs, selected_weights)]) / total_selected
            avg_val_loss = np.sum([l * w for l, w in zip(val_losses, selected_weights)]) / total_selected
            avg_val_acc = np.sum([a * w for a, w in zip(val_accs, selected_weights)]) / total_selected
            
            # Store history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(avg_val_acc)
            
            # Print progress every 10 rounds
            if (round_idx + 1) % 10 == 0:
                print(f"Round {round_idx + 1}/{num_rounds} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        return self.history
    
    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the global model on test data.
        
        Args:
            test_images: Test images
            test_labels: Test labels
            
        Returns:
            Test loss and accuracy
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        test_images = torch.FloatTensor(test_images)
        test_labels = torch.LongTensor(test_labels)
        
        with torch.no_grad():
            outputs = self.global_model(test_images)
            loss = criterion(outputs, test_labels).item()
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == test_labels).float().mean().item()
        
        return loss, accuracy
    
    def save_model(self, path: str):
        """Save the global model to disk."""
        torch.save(self.global_model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load the global model from disk."""
        self.global_model.load_state_dict(torch.load(path))


def load_parallel_federated_data(train_path: str, test_path: str, 
                                train_ratio: float = 0.8) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Load federated data for parallel training with dynamic actor creation.
    
    Args:
        train_path: Path to training data file
        test_path: Path to test data file
        train_ratio: Ratio of local data to use for training
        
    Returns:
        List of client data tuples, test images, test labels
    """
    # Load data
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    
    # Extract test data
    test_images = np.array(test_data[0]['images'])
    test_labels = np.array(test_data[0]['labels'])
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        # Get available resources
        import psutil
        num_cpus = psutil.cpu_count(logical=False) or 4
        
        ray.init(
            ignore_reinit_error=True,
            logging_level='ERROR',
            num_cpus=num_cpus,
            _system_config={
                "automatic_object_spilling_enabled": True,
                "max_io_workers": 4,
            }
        )
        print(f"Ray initialized with {num_cpus} CPUs")
    
    # Prepare client data (without creating actors yet)
    client_data = []
    print(f"Loading data for {len(train_data)} clients...")
    for i in range(len(train_data)):
        images = np.array(train_data[i]['images'])
        labels = np.array(train_data[i]['labels'])
        client_data.append((images, labels))
    
    print(f"Loaded data for {len(client_data)} clients")
    print(f"Test data: {len(test_labels)} samples")
    
    return client_data, test_images, test_labels


def main():
    """
    Main function for Part 2: Parallel Clients with dynamic actor creation.
    """
    print("\n" + "="*80)
    print(" "*20 + "PART 2: PARALLEL CLIENTS")
    print(" "*15 + "Dynamic Ray Actors (4 clients per round)")
    print("="*80 + "\n")
    
    # Load data (no actors created yet)
    client_data, test_images, test_labels = load_parallel_federated_data(
        'Assignment3-data/train_data.npy',
        'Assignment3-data/test_data.npy',
        train_ratio=0.8
    )
    
    # Create federated server
    server = ParallelFederatedServer(client_data, train_ratio=0.8)
    
    # Train with parallel actors (4 clients per round)
    num_rounds = 50
    clients_per_round = 4  # As specified in the requirements
    local_epochs = 10
    
    history = server.train(
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        batch_size=32,
        lr=0.01,
        seed=42
    )
    
    # Evaluate on test data
    test_loss, test_acc = server.evaluate(test_images, test_labels)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS (Part 2)")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Clients per round: {clients_per_round}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Communication rounds: {num_rounds}")
    print(f"  Actor creation: Dynamic (per round)")
    print(f"\nPerformance:")
    print(f"  Final Training Accuracy: {history['train_acc'][-1]:.4f} ({history['train_acc'][-1]*100:.2f}%)")
    print(f"  Final Validation Accuracy: {history['val_acc'][-1]:.4f} ({history['val_acc'][-1]*100:.2f}%)")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"{'='*80}\n")
    
    # Save model
    server.save_model('results/parallel_model.pth')
    print("Model saved to 'results/parallel_model.pth'")
    
    # Shutdown Ray
    ray.shutdown()
    
    return history, test_acc, test_loss


if __name__ == '__main__':
    main()

