"""
Federated Averaging (FedAvg) Implementation for EMNIST Dataset
This implementation follows the FedAvg algorithm for federated learning.
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


class FederatedClient:
    """
    A federated learning client that performs local training on its private data.
    """
    def __init__(self, client_id: int, images: np.ndarray, labels: np.ndarray, 
                 train_ratio: float = 0.8):
        """
        Initialize a federated client.
        
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


@ray.remote
class RayFederatedClient:
    """
    Ray remote actor wrapper for FederatedClient to enable parallel execution.
    """
    def __init__(self, client_id: int, images: np.ndarray, labels: np.ndarray, 
                 train_ratio: float = 0.8):
        self.client = FederatedClient(client_id, images, labels, train_ratio)
    
    def get_data_size(self):
        return self.client.get_data_size()
    
    def local_update(self, global_weights, epochs, batch_size=32, lr=0.01):
        return self.client.local_update(global_weights, epochs, batch_size, lr)


class FederatedServer:
    """
    Federated learning server that coordinates training across clients.
    """
    def __init__(self, clients: List[RayFederatedClient], client_weights: List[int]):
        """
        Initialize the federated server.
        
        Args:
            clients: List of Ray remote client actors
            client_weights: Number of training samples for each client (for weighted averaging)
        """
        self.clients = clients
        self.client_weights = np.array(client_weights)
        self.total_samples = np.sum(self.client_weights)
        self.global_model = FederatedNN()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
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
        Train the global model using FedAvg algorithm.
        
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
        n_clients = len(self.clients)
        
        print(f"\nStarting Federated Training")
        print(f"Number of rounds: {num_rounds}")
        print(f"Clients per round: {clients_per_round}")
        print(f"Local epochs: {local_epochs}")
        print(f"Total clients: {n_clients}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}\n")
        
        for round_idx in tqdm(range(num_rounds), desc="Training Rounds"):
            # Select random clients for this round
            selected_indices = np.random.choice(n_clients, clients_per_round, replace=False)
            selected_clients = [self.clients[i] for i in selected_indices]
            
            # Get current global weights
            global_weights = self.global_model.state_dict()
            
            # Parallel local updates using Ray
            futures = [
                client.local_update.remote(global_weights, local_epochs, batch_size, lr)
                for client in selected_clients
            ]
            results = ray.get(futures)
            
            # Extract local weights and metrics
            local_weights = [result[0] for result in results]
            train_losses = [result[1] for result in results]
            train_accs = [result[2] for result in results]
            val_losses = [result[3] for result in results]
            val_accs = [result[4] for result in results]
            
            # Aggregate weights
            aggregated_weights = self.aggregate_weights(local_weights, selected_indices)
            self.global_model.load_state_dict(aggregated_weights)
            
            # Calculate weighted average metrics based on client data sizes
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


def load_federated_data(train_path: str, test_path: str, 
                       train_ratio: float = 0.8) -> Tuple[List, List[int], np.ndarray, np.ndarray]:
    """
    Load and prepare federated data.
    
    Args:
        train_path: Path to training data file
        test_path: Path to test data file
        train_ratio: Ratio of local data to use for training
        
    Returns:
        List of client actors, list of client weights, test images, test labels
    """
    # Load data
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    
    # Extract test data
    test_images = np.array(test_data[0]['images'])
    test_labels = np.array(test_data[0]['labels'])
    
    # Initialize Ray with reduced logging and resource limits
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            logging_level='ERROR',  # Reduce logging verbosity
            _system_config={
                "automatic_object_spilling_enabled": True,
                "max_io_workers": 4,
            }
        )
    
    # Create client actors
    clients = []
    client_weights = []
    
    print(f"Creating {len(train_data)} federated clients...")
    for i in range(len(train_data)):
        images = np.array(train_data[i]['images'])
        labels = np.array(train_data[i]['labels'])
        
        # Create remote client actor
        client = RayFederatedClient.remote(i, images, labels, train_ratio)
        clients.append(client)
        
        # Get training data size for weighted averaging
        n_train, _ = ray.get(client.get_data_size.remote())
        client_weights.append(n_train)
    
    print(f"Created {len(clients)} clients with {sum(client_weights)} total training samples")
    
    return clients, client_weights, test_images, test_labels


def main():
    """
    Main function to demonstrate FedAvg training.
    """
    # Load data
    clients, client_weights, test_images, test_labels = load_federated_data(
        'Assignment3-data/train_data.npy',
        'Assignment3-data/test_data.npy',
        train_ratio=0.8
    )
    
    # Create federated server
    server = FederatedServer(clients, client_weights)
    
    # Train with FedAvg
    num_rounds = 100
    clients_per_round = max(1, int(0.1 * len(clients)))  # C = 10%
    local_epochs = 5
    
    history = server.train(
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        batch_size=32,
        lr=0.01
    )
    
    # Evaluate on test data
    test_loss, test_acc = server.evaluate(test_images, test_labels)
    print(f"\nTest Results: Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}")
    
    # Save model
    server.save_model('fedavg_model.pth')
    print("Model saved to 'fedavg_model.pth'")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == '__main__':
    main()

