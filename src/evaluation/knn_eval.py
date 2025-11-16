"""
k-Nearest Neighbors (k-NN) evaluation for H-JEPA.

k-NN evaluation is a common protocol for self-supervised learning that measures
feature quality without any training. It classifies test samples based on their
nearest neighbors in the training set.
"""

from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, top_k_accuracy_score


class KNNEvaluator:
    """
    k-Nearest Neighbors evaluator for frozen features.

    Args:
        model: H-JEPA model (will be frozen)
        hierarchy_level: Which hierarchy level to evaluate (0=finest)
        k: Number of neighbors to consider
        distance_metric: Distance metric ('cosine', 'euclidean', 'minkowski')
        pooling: Feature pooling method ('mean', 'max')
        temperature: Temperature for distance weighting (lower = sharper)
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        hierarchy_level: int = 0,
        k: int = 20,
        distance_metric: str = 'cosine',
        pooling: str = 'mean',
        temperature: float = 0.07,
        device: str = 'cuda',
    ):
        self.model = model
        self.hierarchy_level = hierarchy_level
        self.k = k
        self.distance_metric = distance_metric
        self.pooling = pooling
        self.temperature = temperature
        self.device = device

        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Will store training features and labels
        self.train_features = None
        self.train_labels = None
        self.knn_index = None

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool patch features to single vector.

        Args:
            features: Features [B, N, D] or [B, D]

        Returns:
            Pooled features [B, D]
        """
        if features.ndim == 2:
            return features

        if self.pooling == 'mean':
            return features.mean(dim=1)
        elif self.pooling == 'max':
            return features.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        normalize: bool = True,
        desc: str = "Extracting features"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and pool features from dataset.

        Args:
            dataloader: DataLoader for the dataset
            normalize: Whether to L2 normalize features
            desc: Progress bar description

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        all_features = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(self.device)

            # Extract features at specified hierarchy level
            features = self.model.extract_features(
                images,
                level=self.hierarchy_level,
                use_target_encoder=True,
            )

            # Pool features
            features = self.pool_features(features)

            # Normalize if requested
            if normalize:
                features = F.normalize(features, p=2, dim=-1)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, labels

    def build_knn_index(
        self,
        train_loader: DataLoader,
        normalize: bool = True,
    ):
        """
        Build k-NN index from training data.

        Args:
            train_loader: Training data loader
            normalize: Whether to normalize features
        """
        # Extract training features
        self.train_features, self.train_labels = self.extract_features(
            train_loader,
            normalize=normalize,
            desc="Building k-NN index"
        )

        # Build k-NN index
        metric = self.distance_metric
        if metric == 'cosine':
            # For cosine similarity with normalized features, use euclidean
            # distance (equivalent after normalization)
            metric = 'euclidean' if normalize else 'cosine'

        self.knn_index = NearestNeighbors(
            n_neighbors=self.k,
            metric=metric,
            algorithm='auto',
            n_jobs=-1,  # Use all CPU cores
        )
        self.knn_index.fit(self.train_features)

        print(f"Built k-NN index with {len(self.train_features)} samples")

    def predict(
        self,
        test_features: np.ndarray,
        num_classes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels using k-NN.

        Args:
            test_features: Test features [N, D]
            num_classes: Number of classes

        Returns:
            Tuple of (predictions, prediction_probs) [N] and [N, num_classes]
        """
        if self.knn_index is None:
            raise RuntimeError("k-NN index not built. Call build_knn_index first.")

        # Find k nearest neighbors
        distances, indices = self.knn_index.kneighbors(test_features)

        # Get labels of neighbors
        neighbor_labels = self.train_labels[indices]  # [N, k]

        # Weight by distance (closer neighbors have more weight)
        # Convert distances to similarities
        if self.distance_metric == 'cosine' and self.temperature > 0:
            # For cosine: similarity = 1 - distance
            # Then apply softmax with temperature
            similarities = 1 - distances
            weights = np.exp(similarities / self.temperature)
        else:
            # For euclidean: use negative distance
            weights = np.exp(-distances / self.temperature)

        # Normalize weights
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted voting
        prediction_probs = np.zeros((len(test_features), num_classes))

        for i in range(len(test_features)):
            for j in range(self.k):
                label = neighbor_labels[i, j]
                prediction_probs[i, label] += weights[i, j]

        # Get predictions
        predictions = np.argmax(prediction_probs, axis=1)

        return predictions, prediction_probs

    def evaluate(
        self,
        test_loader: DataLoader,
        num_classes: int,
        normalize: bool = True,
        top_k_list: List[int] = [1, 5],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate k-NN classifier on test set.

        Args:
            test_loader: Test data loader
            num_classes: Number of classes
            normalize: Whether to normalize features
            top_k_list: List of k values for top-k accuracy
            verbose: Whether to show progress

        Returns:
            Dictionary with metrics
        """
        if self.knn_index is None:
            raise RuntimeError("k-NN index not built. Call build_knn_index first.")

        # Extract test features
        test_features, test_labels = self.extract_features(
            test_loader,
            normalize=normalize,
            desc="Extracting test features"
        )

        # Predict
        predictions, prediction_probs = self.predict(test_features, num_classes)

        # Compute metrics
        accuracy = accuracy_score(test_labels, predictions) * 100

        metrics = {
            'accuracy': accuracy,
            'top_1_accuracy': accuracy,  # Same as accuracy
        }

        # Top-k accuracies
        for k in top_k_list:
            if k > 1:
                top_k_acc = top_k_accuracy_score(
                    test_labels,
                    prediction_probs,
                    k=min(k, num_classes)
                ) * 100
                metrics[f'top_{k}_accuracy'] = top_k_acc

        if verbose:
            print(f"\nk-NN Evaluation Results (k={self.k}):")
            print(f"  Accuracy: {accuracy:.2f}%")
            for k in top_k_list:
                if k > 1 and f'top_{k}_accuracy' in metrics:
                    print(f"  Top-{k} Accuracy: {metrics[f'top_{k}_accuracy']:.2f}%")

        return metrics

    def evaluate_multiple_k(
        self,
        test_loader: DataLoader,
        num_classes: int,
        k_values: List[int] = [1, 5, 10, 20, 50, 100, 200],
        normalize: bool = True,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate k-NN with different k values.

        Args:
            test_loader: Test data loader
            num_classes: Number of classes
            k_values: List of k values to try
            normalize: Whether to normalize features
            verbose: Whether to show progress

        Returns:
            Dictionary mapping k to metrics
        """
        # Extract test features once
        test_features, test_labels = self.extract_features(
            test_loader,
            normalize=normalize,
            desc="Extracting test features"
        )

        results = {}

        for k in k_values:
            if k > len(self.train_features):
                if verbose:
                    print(f"Skipping k={k} (larger than training set size)")
                continue

            # Update k-NN index with new k
            self.knn_index.n_neighbors = k
            self.k = k

            # Predict
            predictions, prediction_probs = self.predict(test_features, num_classes)

            # Compute accuracy
            accuracy = accuracy_score(test_labels, predictions) * 100

            results[k] = {
                'accuracy': accuracy,
                'top_1_accuracy': accuracy,
            }

            if verbose:
                print(f"k={k:3d}: Accuracy = {accuracy:.2f}%")

        return results


def knn_eval(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    hierarchy_level: int = 0,
    k: int = 20,
    distance_metric: str = 'cosine',
    temperature: float = 0.07,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Convenience function for k-NN evaluation.

    Args:
        model: H-JEPA model
        train_loader: Training data loader
        test_loader: Test data loader
        num_classes: Number of classes
        hierarchy_level: Which hierarchy level to evaluate
        k: Number of neighbors
        distance_metric: Distance metric
        temperature: Temperature for distance weighting
        device: Device to run on
        verbose: Whether to show progress

    Returns:
        Evaluation metrics
    """
    evaluator = KNNEvaluator(
        model=model,
        hierarchy_level=hierarchy_level,
        k=k,
        distance_metric=distance_metric,
        temperature=temperature,
        device=device,
    )

    # Build index
    evaluator.build_knn_index(train_loader)

    # Evaluate
    metrics = evaluator.evaluate(
        test_loader=test_loader,
        num_classes=num_classes,
        verbose=verbose,
    )

    return metrics


def sweep_knn_params(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    hierarchy_level: int = 0,
    k_values: List[int] = [10, 20, 50, 100, 200],
    temperatures: List[float] = [0.01, 0.05, 0.07, 0.1, 0.5],
    distance_metrics: List[str] = ['cosine', 'euclidean'],
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Sweep over k-NN hyperparameters to find best configuration.

    Args:
        model: H-JEPA model
        train_loader: Training data loader
        test_loader: Test data loader
        num_classes: Number of classes
        hierarchy_level: Which hierarchy level to evaluate
        k_values: List of k values to try
        temperatures: List of temperatures to try
        distance_metrics: List of distance metrics to try
        device: Device to run on

    Returns:
        Dictionary with results for each configuration
    """
    results = {}
    best_acc = 0.0
    best_config = None

    print("Sweeping k-NN hyperparameters...")
    print(f"k values: {k_values}")
    print(f"Temperatures: {temperatures}")
    print(f"Distance metrics: {distance_metrics}")
    print()

    for metric in distance_metrics:
        for temp in temperatures:
            for k in k_values:
                config_name = f"{metric}_k{k}_t{temp}"

                print(f"Testing {config_name}...")

                try:
                    evaluator = KNNEvaluator(
                        model=model,
                        hierarchy_level=hierarchy_level,
                        k=k,
                        distance_metric=metric,
                        temperature=temp,
                        device=device,
                    )

                    evaluator.build_knn_index(train_loader)
                    metrics = evaluator.evaluate(
                        test_loader=test_loader,
                        num_classes=num_classes,
                        verbose=False,
                    )

                    results[config_name] = {
                        'config': {
                            'k': k,
                            'temperature': temp,
                            'distance_metric': metric,
                        },
                        'metrics': metrics,
                    }

                    acc = metrics['accuracy']
                    print(f"  Accuracy: {acc:.2f}%")

                    if acc > best_acc:
                        best_acc = acc
                        best_config = config_name

                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue

    print(f"\nBest configuration: {best_config}")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Config: {results[best_config]['config']}")

    return results
