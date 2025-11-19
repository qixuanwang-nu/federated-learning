"""
Part 3: Federated Learning with Differential Privacy (Laplace Mechanism)
Implements FedAvg with Laplace noise injection for privacy protection.
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
    """
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=62):
        super(FederatedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def add_laplace_noise(images: np.ndarray, noise_scale: float) -> np.ndarray:
    """
    Add Laplace noise to images for differential privacy.
    
    For each 28×28 image X, add noise ε ~ Lap(0, b) to get X' = X + ε
    where each ε_ij is sampled independently from Laplace(0, b).
    
    Args:
        images: Input images, shape (n_samples, 28, 28)
        noise_scale: Scale parameter b for Laplace distribution
        
    Returns:
        Perturbed images with Laplace noise added
    """
    if noise_scale == 0:
        return images.copy()
    
    # Sample noise from Laplace distribution
    # numpy.random.laplace(loc=0, scale=b, size=shape)
    noise = np.random.laplace(loc=0.0, scale=noise_scale, size=images.shape)
    
    # Add noise to images
    perturbed_images = images + noise
    
    # Clip to valid range [0, 1] since original images are in [0, 1]
    perturbed_images = np.clip(perturbed_images, 0.0, 1.0)
    
    return perturbed_images


@ray.remote
class DPFederatedClient:
    """
    Federated client with differential privacy via Laplace mechanism.
    """
    def __init__(self, client_id: int, images: np.ndarray, labels: np.ndarray,
                 noise_scale: float, train_ratio: float = 0.8):
        """
        Initialize a DP federated client.
        
        Args:
            client_id: Unique identifier
            images: Local images (will be perturbed)
            labels: Local labels (not perturbed)
            noise_scale: Laplace noise scale b
            train_ratio: Train/validation split ratio
        """
        self.client_id = client_id
        self.noise_scale = noise_scale
        
        # Split data into train and validation
        n_samples = len(images)
        n_train = int(n_samples * train_ratio)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Add Laplace noise to training images for privacy
        train_images = images[train_indices]
        val_images = images[val_indices]
        
        # Perturb training data with Laplace noise
        train_images_perturbed = add_laplace_noise(train_images, noise_scale)
        
        # Validation data is also perturbed (consistent with training)
        val_images_perturbed = add_laplace_noise(val_images, noise_scale)
        
        # Convert to PyTorch tensors
        self.train_images = torch.FloatTensor(train_images_perturbed)
        self.train_labels = torch.LongTensor(labels[train_indices])
        self.val_images = torch.FloatTensor(val_images_perturbed)
        self.val_labels = torch.LongTensor(labels[val_indices])
        
        self.n_train = len(self.train_labels)
        self.n_val = len(self.val_labels)
    
    def get_data_size(self) -> Tuple[int, int]:
        return self.n_train, self.n_val
    
    def local_update(self, global_weights: Dict, epochs: int,
                    batch_size: int = 32, lr: float = 0.01) -> Tuple[Dict, float, float, float, float]:
        """
        Perform local training on perturbed data.
        """
        model = FederatedNN()
        model.load_state_dict(global_weights)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_dataset = TensorDataset(self.train_images, self.train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Local training on perturbed data
        for epoch in range(epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate on perturbed data
        train_loss, train_acc = self._evaluate(model, self.train_images, self.train_labels, criterion)
        val_loss, val_acc = self._evaluate(model, self.val_images, self.val_labels, criterion)
        
        return model.state_dict(), train_loss, train_acc, val_loss, val_acc
    
    def _evaluate(self, model: nn.Module, images: torch.Tensor,
                 labels: torch.Tensor, criterion: nn.Module) -> Tuple[float, float]:
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels).item()
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == labels).float().mean().item()
        model.train()
        return loss, accuracy


class DPFederatedServer:
    """
    Federated server for training with differential privacy.
    """
    def __init__(self, clients: List, client_weights: List[int]):
        self.clients = clients
        self.client_weights = np.array(client_weights)
        self.total_samples = np.sum(self.client_weights)
        self.global_model = FederatedNN()
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def aggregate_weights(self, local_weights: List[Dict],
                         selected_indices: List[int]) -> Dict:
        selected_weights = self.client_weights[selected_indices]
        total_selected = np.sum(selected_weights)
        
        global_weights = copy.deepcopy(local_weights[0])
        for key in global_weights.keys():
            global_weights[key] = torch.zeros_like(global_weights[key])
        
        for local_w, weight in zip(local_weights, selected_weights):
            for key in global_weights.keys():
                global_weights[key] += local_w[key] * (weight / total_selected)
        
        return global_weights
    
    def train(self, num_rounds: int, clients_per_round: int,
             local_epochs: int, batch_size: int = 32, lr: float = 0.01,
             seed: int = 42, verbose: bool = True) -> Dict:
        """
        Train with FedAvg using DP clients.
        """
        np.random.seed(seed)
        n_clients = len(self.clients)
        
        if verbose:
            print(f"\nStarting DP Federated Training")
            print(f"Rounds: {num_rounds}, Clients/round: {clients_per_round}, "
                  f"Local epochs: {local_epochs}")
        
        for round_idx in tqdm(range(num_rounds), desc="Training", disable=not verbose):
            selected_indices = np.random.choice(n_clients, clients_per_round, replace=False)
            selected_clients = [self.clients[i] for i in selected_indices]
            
            global_weights = self.global_model.state_dict()
            
            futures = [
                client.local_update.remote(global_weights, local_epochs, batch_size, lr)
                for client in selected_clients
            ]
            results = ray.get(futures)
            
            local_weights = [result[0] for result in results]
            train_losses = [result[1] for result in results]
            train_accs = [result[2] for result in results]
            val_losses = [result[3] for result in results]
            val_accs = [result[4] for result in results]
            
            aggregated_weights = self.aggregate_weights(local_weights, selected_indices)
            self.global_model.load_state_dict(aggregated_weights)
            
            selected_weights = self.client_weights[selected_indices]
            total_selected = np.sum(selected_weights)
            
            avg_train_loss = np.sum([l * w for l, w in zip(train_losses, selected_weights)]) / total_selected
            avg_train_acc = np.sum([a * w for a, w in zip(train_accs, selected_weights)]) / total_selected
            avg_val_loss = np.sum([l * w for l, w in zip(val_losses, selected_weights)]) / total_selected
            avg_val_acc = np.sum([a * w for a, w in zip(val_accs, selected_weights)]) / total_selected
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(avg_val_acc)
            
            if verbose and (round_idx + 1) % 10 == 0:
                print(f"Round {round_idx + 1}/{num_rounds} - "
                      f"Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        return self.history
    
    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray,
                noise_scale: float = 0.0) -> Tuple[float, float]:
        """
        Evaluate on test data.
        
        Args:
            test_images: Clean test images
            test_labels: Test labels
            noise_scale: If >0, add noise to test images (for consistency)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        # Optionally perturb test data
        if noise_scale > 0:
            test_images = add_laplace_noise(test_images, noise_scale)
        
        test_images = torch.FloatTensor(test_images)
        test_labels = torch.LongTensor(test_labels)
        
        with torch.no_grad():
            outputs = self.global_model(test_images)
            loss = criterion(outputs, test_labels).item()
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == test_labels).float().mean().item()
        
        return loss, accuracy
    
    def save_model(self, path: str):
        torch.save(self.global_model.state_dict(), path)


def load_dp_federated_data(train_path: str, test_path: str,
                          noise_scale: float, train_ratio: float = 0.8):
    """
    Load data and create DP federated clients.
    """
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    
    test_images = np.array(test_data[0]['images'])
    test_labels = np.array(test_data[0]['labels'])
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level='ERROR')
    
    clients = []
    client_weights = []
    
    for i in range(len(train_data)):
        images = np.array(train_data[i]['images'])
        labels = np.array(train_data[i]['labels'])
        
        client = DPFederatedClient.remote(i, images, labels, noise_scale, train_ratio)
        clients.append(client)
        
        n_train, _ = ray.get(client.get_data_size.remote())
        client_weights.append(n_train)
    
    return clients, client_weights, test_images, test_labels


def main():
    """
    Demo with noise_scale = 0.1
    """
    print("\n" + "="*80)
    print(" "*20 + "PART 3: DIFFERENTIAL PRIVACY")
    print(" "*15 + "FedAvg with Laplace Mechanism")
    print("="*80 + "\n")
    
    noise_scale = 0.1
    
    clients, client_weights, test_images, test_labels = load_dp_federated_data(
        'Assignment3-data/train_data.npy',
        'Assignment3-data/test_data.npy',
        noise_scale=noise_scale,
        train_ratio=0.8
    )
    
    server = DPFederatedServer(clients, client_weights)
    
    history = server.train(
        num_rounds=50,
        clients_per_round=max(1, int(0.1 * len(clients))),
        local_epochs=5,
        batch_size=32,
        lr=0.01,
        seed=42
    )
    
    # Evaluate on clean test data
    test_loss, test_acc = server.evaluate(test_images, test_labels, noise_scale=0.0)
    
    print(f"\n{'='*80}")
    print(f"Results with noise_scale = {noise_scale}")
    print(f"{'='*80}")
    print(f"Final Train Acc: {history['train_acc'][-1]:.4f}")
    print(f"Final Val Acc: {history['val_acc'][-1]:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"{'='*80}\n")
    
    ray.shutdown()


if __name__ == '__main__':
    main()

