"""
Linear probe evaluation for H-JEPA.

This module implements linear probe evaluation, which trains a linear classifier
on frozen features from H-JEPA. This is a standard evaluation protocol for
self-supervised learning models.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm


class LinearProbe(nn.Module):
    """
    Linear classifier head for frozen feature evaluation.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        pooling: Feature pooling method ('mean', 'cls', 'max', 'attention')
        normalize: Whether to L2 normalize features before classification
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pooling: str = "mean",
        normalize: bool = False,
    ):
        super().__init__()

        if pooling not in ["mean", "cls", "max", "attention"]:
            raise ValueError(f"Invalid pooling method: {pooling}")

        self.pooling = pooling
        self.normalize = normalize

        # Linear classifier
        self.classifier = nn.Linear(input_dim, num_classes)

        # Attention pooling (if needed)
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, 1),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool patch features to single vector.

        Args:
            features: Features [B, N, D] or [B, D]

        Returns:
            Pooled features [B, D]
        """
        if features.ndim == 2:
            # Already pooled
            return features

        if self.pooling == "mean":
            # Mean pooling over all patches
            pooled = features.mean(dim=1)
        elif self.pooling == "cls":
            # Use first token (CLS token) - assuming it's present in features
            # Note: extract_features removes CLS, so this won't work by default
            # We'll fall back to mean pooling
            warnings.warn(
                "CLS pooling requested but features don't have CLS token. Using mean pooling."
            )
            pooled = features.mean(dim=1)
        elif self.pooling == "max":
            # Max pooling over patches
            pooled = features.max(dim=1)[0]
        elif self.pooling == "attention":
            # Attention-weighted pooling
            attention_weights = self.attention(features)  # [B, N, 1]
            attention_weights = F.softmax(attention_weights, dim=1)
            pooled = (features * attention_weights).sum(dim=1)

        return pooled

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Input features [B, N, D] or [B, D]

        Returns:
            Logits [B, num_classes]
        """
        # Pool features
        pooled = self.pool_features(features)

        # Normalize if requested
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        # Classify
        logits = self.classifier(pooled)

        return logits


class LinearProbeEvaluator:
    """
    Evaluator for linear probe protocol.

    Args:
        model: H-JEPA model (will be frozen)
        num_classes: Number of classes in the dataset
        input_dim: Feature dimension
        hierarchy_level: Which hierarchy level to evaluate (0=finest)
        pooling: Feature pooling method
        normalize: Whether to normalize features
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        input_dim: int,
        hierarchy_level: int = 0,
        pooling: str = "mean",
        normalize: bool = False,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.hierarchy_level = hierarchy_level

        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Create linear probe
        self.probe = LinearProbe(
            input_dim=input_dim,
            num_classes=num_classes,
            pooling=pooling,
            normalize=normalize,
        ).to(device)

    @torch.no_grad()
    def extract_features(
        self, dataloader: DataLoader, desc: str = "Extracting features"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataset.

        Args:
            dataloader: DataLoader for the dataset
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

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, labels

    def train_probe(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        lr: float = 0.1,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        scheduler_type: str = "cosine",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train linear probe on frozen features.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay
            momentum: SGD momentum
            scheduler_type: LR scheduler type ('cosine', 'step', None)
            verbose: Whether to show progress

        Returns:
            Dictionary with training history
        """
        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.probe.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Setup scheduler
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
        else:
            scheduler = None

        criterion = nn.CrossEntropyLoss()

        # Training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            # Training
            self.probe.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader

            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # Extract frozen features
                with torch.no_grad():
                    features = self.model.extract_features(
                        images,
                        level=self.hierarchy_level,
                        use_target_encoder=True,
                    )

                # Forward through probe
                logits = self.probe(features)
                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                if verbose:
                    pbar.set_postfix(
                        {
                            "loss": train_loss / (pbar.n + 1),
                            "acc": 100.0 * train_correct / train_total,
                        }
                    )

            train_acc = 100.0 * train_correct / train_total
            train_loss = train_loss / len(train_loader)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, verbose=False)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])

                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                        f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                    )

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

        return history

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_confusion: bool = False,
        top_k: int = 5,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate linear probe.

        Args:
            dataloader: Evaluation data loader
            compute_confusion: Whether to compute confusion matrix
            top_k: k for top-k accuracy
            verbose: Whether to show progress

        Returns:
            Dictionary with metrics
        """
        self.probe.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Extract frozen features
            features = self.model.extract_features(
                images,
                level=self.hierarchy_level,
                use_target_encoder=True,
            )

            # Forward
            logits = self.probe(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            # Get predictions
            probs = F.softmax(logits, dim=-1)
            _, predicted = logits.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        # Concatenate results
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100

        # Top-k accuracy
        top_k_acc = (
            top_k_accuracy_score(all_labels, all_probs, k=min(top_k, all_probs.shape[1])) * 100
        )

        avg_loss = total_loss / len(dataloader)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            f"top_{top_k}_accuracy": top_k_acc,
        }

        # Confusion matrix
        if compute_confusion:
            conf_matrix = confusion_matrix(all_labels, all_preds)
            metrics["confusion_matrix"] = conf_matrix

        return metrics

    def k_fold_cross_validation(
        self,
        dataset: torch.utils.data.Dataset,
        k_folds: int = 5,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.1,
        num_workers: int = 4,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.

        Args:
            dataset: Dataset to split into folds
            k_folds: Number of folds
            epochs: Training epochs per fold
            batch_size: Batch size
            lr: Learning rate
            num_workers: Data loading workers
            verbose: Whether to show progress

        Returns:
            Dictionary with results for each fold
        """
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        results = {
            "fold_accuracies": [],
            "fold_losses": [],
            "mean_accuracy": 0.0,
            "std_accuracy": 0.0,
        }

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Fold {fold + 1}/{k_folds}")
                print(f"{'='*50}")

            # Create data samplers
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)

            # Create data loaders
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )

            # Reset probe weights
            self.probe._init_weights()

            # Train
            self.train_probe(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                verbose=verbose,
            )

            # Evaluate
            metrics = self.evaluate(val_loader, verbose=False)

            results["fold_accuracies"].append(metrics["accuracy"])
            results["fold_losses"].append(metrics["loss"])

            if verbose:
                print(f"Fold {fold + 1} - Accuracy: {metrics['accuracy']:.2f}%")

        # Compute statistics
        results["mean_accuracy"] = np.mean(results["fold_accuracies"])
        results["std_accuracy"] = np.std(results["fold_accuracies"])

        if verbose:
            print(f"\n{'='*50}")
            print(f"Cross-Validation Results")
            print(f"{'='*50}")
            print(
                f"Mean Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%"
            )
            print(f"Fold Accuracies: {results['fold_accuracies']}")

        return results


def linear_probe_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    hierarchy_level: int = 0,
    epochs: int = 100,
    lr: float = 0.1,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Convenience function for linear probe evaluation.

    Args:
        model: H-JEPA model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        hierarchy_level: Which hierarchy level to evaluate
        epochs: Training epochs
        lr: Learning rate
        device: Device to run on
        verbose: Whether to show progress

    Returns:
        Evaluation metrics
    """
    # Get input dimension from model
    input_dim = model.embed_dim

    # Create evaluator
    evaluator = LinearProbeEvaluator(
        model=model,
        num_classes=num_classes,
        input_dim=input_dim,
        hierarchy_level=hierarchy_level,
        device=device,
    )

    # Train
    evaluator.train_probe(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        verbose=verbose,
    )

    # Evaluate
    metrics = evaluator.evaluate(
        dataloader=val_loader,
        compute_confusion=True,
        verbose=verbose,
    )

    return metrics
