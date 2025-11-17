"""
Transfer learning evaluation for H-JEPA.

This module implements transfer learning protocols including fine-tuning,
few-shot learning, and domain adaptation evaluation.
"""

import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TransferHead(nn.Module):
    """
    Classification head for transfer learning.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions (empty for linear)
        dropout: Dropout probability
        pooling: Feature pooling method
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [],
        dropout: float = 0.0,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling

        # Build MLP
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pool patch features if needed."""
        if features.ndim == 2:
            return features

        if self.pooling == "mean":
            return features.mean(dim=1)
        elif self.pooling == "max":
            return features.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.pool_features(features)
        logits: torch.Tensor = self.mlp(features)
        return logits


class FineTuneEvaluator:
    """
    Fine-tuning evaluator for transfer learning.

    Args:
        model: H-JEPA model
        num_classes: Number of classes in target dataset
        hierarchy_level: Which hierarchy level to use
        freeze_encoder: Whether to freeze encoder during fine-tuning
        hidden_dims: Hidden dimensions for classification head
        dropout: Dropout probability
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        hierarchy_level: int = 0,
        freeze_encoder: bool = False,
        hidden_dims: List[int] = [],
        dropout: float = 0.0,
        device: str = "cuda",
    ):
        self.device = device
        self.hierarchy_level = hierarchy_level
        self.freeze_encoder = freeze_encoder

        # Clone model to avoid modifying original
        self.model = copy.deepcopy(model).to(device)

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        # Create classification head
        input_dim: int = int(self.model.embed_dim)  # type: ignore[arg-type]
        self.classifier = TransferHead(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and classifier."""
        # Extract features
        features: torch.Tensor
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.model.extract_features(  # type: ignore[operator]
                    images,
                    level=self.hierarchy_level,
                    use_target_encoder=True,
                )
        else:
            # Use context encoder for fine-tuning (has gradients)
            features = self.model.extract_features(  # type: ignore[operator]
                images,
                level=self.hierarchy_level,
                use_target_encoder=False,
            )

        # Classify
        logits: torch.Tensor = self.classifier(features)
        return logits

    def train_epoch(
        self,
        train_loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train() if not self.freeze_encoder else self.model.eval()
        self.classifier.train()

        total_loss = 0.0
        correct = 0.0
        total = 0

        pbar: Union[tqdm[Any], DataLoader[Any]] = (
            tqdm(train_loader, desc=f"Epoch {epoch}") if verbose else train_loader
        )

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward
            logits = self.forward(images)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix(
                    {"loss": total_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
                )

        metrics = {
            "loss": total_loss / len(train_loader),
            "accuracy": 100.0 * correct / total,
        }

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader[Any],
        criterion: nn.Module,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        self.classifier.eval()

        total_loss = 0.0
        correct = 0.0
        total = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(test_loader, desc="Evaluating") if verbose else test_loader

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward
            logits = self.forward(images)
            loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        metrics = {
            "loss": total_loss / len(test_loader),
            "accuracy": 100.0 * correct / total,
        }

        return metrics

    def fine_tune(
        self,
        train_loader: DataLoader[Any],
        val_loader: Optional[DataLoader[Any]] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune model on target dataset.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            scheduler_type: LR scheduler type
            verbose: Whether to show progress

        Returns:
            Training history
        """
        # Setup optimizer
        params: Union[Any, List[Dict[str, Any]]]
        if self.freeze_encoder:
            # Only optimize classifier
            params = self.classifier.parameters()
        else:
            # Optimize both encoder and classifier (with different LRs)
            params = [
                {"params": self.model.parameters(), "lr": lr * 0.1},
                {"params": self.classifier.parameters(), "lr": lr},
            ]

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Setup scheduler
        scheduler: Optional[
            Union[torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.StepLR]
        ] = None
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)

        criterion = nn.CrossEntropyLoss()

        # Training history
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch, verbose)

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, criterion, verbose=False)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])

                if verbose:
                    print(
                        f"Epoch {epoch}/{epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.2f}% - "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.2f}%"
                    )

                # Track best
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
            else:
                if verbose:
                    print(
                        f"Epoch {epoch}/{epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.2f}%"
                    )

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

        if val_loader is not None and verbose:
            print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")

        return history


class FewShotEvaluator:
    """
    Few-shot learning evaluator.

    Evaluates the model's ability to learn from very few examples.

    Args:
        model: H-JEPA model
        num_classes: Number of classes
        hierarchy_level: Which hierarchy level to use
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        hierarchy_level: int = 0,
        device: str = "cuda",
    ):
        self.model = model
        self.num_classes = num_classes
        self.hierarchy_level = hierarchy_level
        self.device = device

        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def sample_few_shot_episodes(
        self,
        dataset: Dataset[Any],
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int,
    ) -> List[Dict[str, Any]]:
        """
        Sample few-shot learning episodes.

        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per episode
            k_shot: Number of examples per class (support set)
            n_query: Number of query examples per class
            n_episodes: Number of episodes to sample

        Returns:
            List of episode dictionaries
        """
        # Get all labels
        all_labels_list: List[Any] = []
        for i in range(len(dataset)):  # type: ignore[arg-type]
            _, label = dataset[i]
            all_labels_list.append(label)
        all_labels: npt.NDArray[np.int64] = np.array(all_labels_list)

        # Get indices for each class
        class_indices: Dict[int, npt.NDArray[np.int64]] = {}
        for class_idx in range(self.num_classes):
            class_indices[class_idx] = np.where(all_labels == class_idx)[0]

        episodes: List[Dict[str, Any]] = []

        for _ in range(n_episodes):
            # Sample n_way classes
            selected_classes: npt.NDArray[np.int64] = np.random.choice(
                self.num_classes, size=n_way, replace=False
            )

            support_indices: List[Any] = []
            query_indices: List[Any] = []

            for class_idx in selected_classes:
                # Sample k_shot + n_query examples
                indices = class_indices[class_idx]
                sampled = np.random.choice(indices, size=k_shot + n_query, replace=False)

                support_indices.extend(sampled[:k_shot])
                query_indices.extend(sampled[k_shot:])

            episodes.append(
                {
                    "support_indices": support_indices,
                    "query_indices": query_indices,
                    "classes": selected_classes,
                }
            )

        return episodes

    @torch.no_grad()
    def evaluate_episode(
        self,
        dataset: Dataset[Any],
        episode: Dict[str, Any],
        metric: str = "cosine",
    ) -> float:
        """
        Evaluate one few-shot episode using nearest centroid classifier.

        Args:
            dataset: Dataset
            episode: Episode dictionary
            metric: Distance metric

        Returns:
            Accuracy on this episode
        """
        # Extract support features
        support_features_list: List[torch.Tensor] = []
        support_labels_list: List[Any] = []

        for idx in episode["support_indices"]:
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(self.device)

            features: torch.Tensor = self.model.extract_features(  # type: ignore[operator]
                image, level=self.hierarchy_level, use_target_encoder=True
            )
            features = features.mean(dim=1)  # Pool
            features = F.normalize(features, p=2, dim=-1)

            support_features_list.append(features)
            support_labels_list.append(label)

        support_features: torch.Tensor = torch.cat(support_features_list, dim=0)
        support_labels: torch.Tensor = torch.tensor(support_labels_list)

        # Compute class centroids
        class_to_new_label = {c: i for i, c in enumerate(episode["classes"])}
        n_way = len(episode["classes"])

        centroids_list: List[torch.Tensor] = []
        for class_idx in episode["classes"]:
            mask = support_labels == class_idx
            centroid = support_features[mask].mean(dim=0)
            centroids_list.append(centroid)

        centroids: torch.Tensor = torch.stack(centroids_list)  # [n_way, D]

        # Evaluate on query set
        correct = 0
        total = 0

        for idx in episode["query_indices"]:
            image, true_label = dataset[idx]
            image = image.unsqueeze(0).to(self.device)

            features_query: torch.Tensor = self.model.extract_features(  # type: ignore[operator]
                image, level=self.hierarchy_level, use_target_encoder=True
            )
            features_query = features_query.mean(dim=1)
            features_query = F.normalize(features_query, p=2, dim=-1)

            # Compute similarity to centroids
            predicted_idx: int
            if metric == "cosine":
                similarities = features_query @ centroids.T
                predicted_idx = int(similarities.argmax(dim=-1).item())
            else:
                distances = torch.cdist(features_query, centroids.unsqueeze(0)).squeeze(0)
                predicted_idx = int(distances.argmin(dim=-1).item())

            # Map back to original label
            predicted_label = episode["classes"][predicted_idx]

            if predicted_label == true_label:
                correct += 1
            total += 1

        accuracy = 100.0 * correct / total
        return accuracy

    def evaluate_few_shot(
        self,
        dataset: Dataset[Any],
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        n_episodes: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate few-shot learning performance.

        Args:
            dataset: Dataset
            n_way: Number of classes per episode
            k_shot: Number of examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes
            verbose: Whether to show progress

        Returns:
            Dictionary with metrics
        """
        if verbose:
            print(f"\nEvaluating {n_way}-way {k_shot}-shot learning...")

        # Sample episodes
        episodes = self.sample_few_shot_episodes(dataset, n_way, k_shot, n_query, n_episodes)

        # Evaluate each episode
        accuracies = []

        pbar = tqdm(episodes, desc="Evaluating episodes") if verbose else episodes

        for episode in pbar:
            acc = self.evaluate_episode(dataset, episode)
            accuracies.append(acc)

        # Compute statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        conf_interval = 1.96 * std_acc / np.sqrt(n_episodes)

        metrics = {
            "accuracy": mean_acc,
            "std": std_acc,
            "confidence_interval": conf_interval,
            "n_way": n_way,
            "k_shot": k_shot,
        }

        if verbose:
            print(f"{n_way}-way {k_shot}-shot: " f"{mean_acc:.2f}% ± {conf_interval:.2f}%")

        return metrics


def fine_tune_eval(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    num_classes: int,
    hierarchy_level: int = 0,
    freeze_encoder: bool = False,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Convenience function for fine-tuning evaluation.

    Args:
        model: H-JEPA model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        hierarchy_level: Which hierarchy level to use
        freeze_encoder: Whether to freeze encoder
        epochs: Number of epochs
        lr: Learning rate
        device: Device to run on
        verbose: Whether to show progress

    Returns:
        Evaluation metrics
    """
    evaluator = FineTuneEvaluator(
        model=model,
        num_classes=num_classes,
        hierarchy_level=hierarchy_level,
        freeze_encoder=freeze_encoder,
        device=device,
    )

    # Fine-tune
    history = evaluator.fine_tune(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        verbose=verbose,
    )

    # Final evaluation
    criterion = nn.CrossEntropyLoss()
    final_metrics = evaluator.evaluate(val_loader, criterion, verbose=False)

    return final_metrics


def few_shot_eval(
    model: nn.Module,
    dataset: Dataset[Any],
    num_classes: int,
    n_way: int = 5,
    k_shot_list: List[int] = [1, 5, 10],
    n_episodes: int = 100,
    hierarchy_level: int = 0,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Convenience function for few-shot evaluation.

    Args:
        model: H-JEPA model
        dataset: Dataset
        num_classes: Number of classes
        n_way: Number of classes per episode
        k_shot_list: List of k-shot values to evaluate
        n_episodes: Number of episodes
        hierarchy_level: Which hierarchy level to use
        device: Device to run on
        verbose: Whether to show progress

    Returns:
        Dictionary mapping k-shot to metrics
    """
    evaluator = FewShotEvaluator(
        model=model,
        num_classes=num_classes,
        hierarchy_level=hierarchy_level,
        device=device,
    )

    results = {}

    for k_shot in k_shot_list:
        metrics = evaluator.evaluate_few_shot(
            dataset=dataset,
            n_way=n_way,
            k_shot=k_shot,
            n_episodes=n_episodes,
            verbose=verbose,
        )
        results[k_shot] = metrics

    return results
