#!/usr/bin/env python3
"""
Evaluation script for H-JEPA models.
Tests the quality of learned representations through:
1. Linear probe evaluation
2. KNN evaluation
3. Visualization of learned features
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import build_dataset
from src.models.hjepa import HJEPA


class LinearProbe(nn.Module):
    """Simple linear classifier for evaluation."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def extract_features(model, dataloader, device, desc="Extracting features"):
    """Extract features from the encoder."""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            # Handle both tuple and dict formats
            if isinstance(batch, (list, tuple)):
                images, batch_labels = batch
                images = images.to(device)
            else:
                images = batch["image"].to(device)
                batch_labels = batch["label"]

            # Get encoder features
            # HJEPA uses context_encoder for inference
            encoder_output = model.context_encoder(images)

            # Handle different output formats
            if hasattr(encoder_output, "last_hidden_state"):
                # Transformer-like output with attributes
                feat = encoder_output.last_hidden_state[:, 0]  # CLS token
            elif encoder_output.dim() == 3:
                # Sequence output [batch_size, num_tokens, embed_dim]
                # Extract CLS token (first position)
                feat = encoder_output[:, 0]
            elif encoder_output.dim() == 4:
                # CNN-like output [batch_size, channels, height, width]
                feat = encoder_output.mean(dim=(2, 3))
            else:
                # Already flattened
                feat = encoder_output

            features.append(feat.cpu())
            labels.append(batch_labels)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels


def evaluate_linear_probe(
    model, train_loader, val_loader, device, num_classes=10, epochs=100, lr=0.01
):
    """Evaluate with linear probe."""
    print("\n" + "=" * 50)
    print("Linear Probe Evaluation")
    print("=" * 50)

    # Extract features
    train_features, train_labels = extract_features(
        model, train_loader, device, "Extracting train features"
    )
    val_features, val_labels = extract_features(
        model, val_loader, device, "Extracting val features"
    )

    # Get feature dimension
    feature_dim = train_features.shape[1]
    print(f"Feature dimension: {feature_dim}")

    # Create linear probe
    probe = LinearProbe(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Move features back to device for training
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    # Train linear probe
    best_acc = 0.0
    for epoch in range(epochs):
        probe.train()

        # Simple batch training
        batch_size = 256
        num_batches = len(train_features) // batch_size

        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(train_features))

            batch_features = train_features[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = probe(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            probe.eval()
            with torch.no_grad():
                outputs = probe(val_features)
                _, predicted = outputs.max(1)
                accuracy = (predicted == val_labels).float().mean().item() * 100

                if accuracy > best_acc:
                    best_acc = accuracy

                print(
                    f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/num_batches:.4f}, Val Acc: {accuracy:.2f}%"
                )

    print(f"\nBest Linear Probe Accuracy: {best_acc:.2f}%")
    return best_acc


def evaluate_knn(model, train_loader, val_loader, device, k=20, temperature=0.07):
    """Evaluate with KNN classifier."""
    print("\n" + "=" * 50)
    print("KNN Evaluation")
    print("=" * 50)

    # Extract features
    train_features, train_labels = extract_features(
        model, train_loader, device, "Extracting train features"
    )
    val_features, val_labels = extract_features(
        model, val_loader, device, "Extracting val features"
    )

    # Normalize features
    train_features = torch.nn.functional.normalize(train_features, dim=1)
    val_features = torch.nn.functional.normalize(val_features, dim=1)

    # Convert to numpy
    train_features = train_features.numpy()
    val_features = val_features.numpy()
    train_labels = train_labels.numpy()
    val_labels = val_labels.numpy()

    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_features, train_labels)

    # Predict
    predictions = knn.predict(val_features)
    accuracy = (predictions == val_labels).mean() * 100

    print(f"KNN (k={k}) Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate H-JEPA model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument(
        "--linear_probe_epochs", type=int, default=100, help="Number of epochs for linear probe"
    )
    parser.add_argument(
        "--linear_probe_lr", type=float, default=0.01, help="Learning rate for linear probe"
    )
    parser.add_argument("--knn_k", type=int, default=20, help="K for KNN evaluation")

    args = parser.parse_args()

    # Load config
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update data path if provided
    if args.data_path:
        config["data"]["data_path"] = args.data_path

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build dataset
    print("\nBuilding datasets...")
    data_config = config["data"]
    val_dataset = build_dataset(
        dataset_name=data_config["dataset"],
        data_path=data_config.get("data_path", "./data"),
        split="val",
        image_size=data_config.get("image_size", 224),
    )

    # Create data loaders
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False
    )

    # For linear probe and KNN, we also need train set
    train_dataset = build_dataset(
        dataset_name=data_config["dataset"],
        data_path=data_config.get("data_path", "./data"),
        split="train",
        image_size=data_config.get("image_size", 224),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False
    )

    # Build model
    print("\nBuilding model...")
    model_config = config["model"]

    # Filter config to only include parameters accepted by HJEPA
    import inspect

    valid_params = inspect.signature(HJEPA.__init__).parameters.keys()
    filtered_config = {k: v for k, v in model_config.items() if k in valid_params}

    # Log filtered parameters
    excluded_params = set(model_config.keys()) - set(filtered_config.keys())
    if excluded_params:
        print(f"Note: Ignoring config parameters not accepted by HJEPA: {excluded_params}")

    model = HJEPA(**filtered_config).to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully!")

    # Get number of classes
    num_classes = len(train_dataset.classes) if hasattr(train_dataset, "classes") else 10

    # Evaluate
    results = {}

    # Linear probe evaluation
    linear_acc = evaluate_linear_probe(
        model,
        train_loader,
        val_loader,
        device,
        num_classes=num_classes,
        epochs=args.linear_probe_epochs,
        lr=args.linear_probe_lr,
    )
    results["linear_probe_accuracy"] = linear_acc

    # KNN evaluation
    knn_acc = evaluate_knn(model, train_loader, val_loader, device, k=args.knn_k)
    results["knn_accuracy"] = knn_acc

    # Save results
    checkpoint_dir = Path(args.checkpoint).parent
    results_file = checkpoint_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Linear Probe Accuracy: {linear_acc:.2f}%")
    print(f"KNN Accuracy (k={args.knn_k}): {knn_acc:.2f}%")

    # Interpretation
    print("\n" + "=" * 50)
    print("Interpretation")
    print("=" * 50)

    if linear_acc >= 70:
        print("✓ Good representation quality! The model has learned meaningful features.")
        if linear_acc >= 80:
            print("✓ Excellent! The representations are highly discriminative.")
    elif linear_acc >= 50:
        print(
            "→ Moderate representation quality. The model is learning but may need more training."
        )
    else:
        print("✗ Poor representation quality. Consider:")
        print("  - Training for more epochs")
        print("  - Adjusting learning rate")
        print("  - Checking loss convergence")
        print("  - Verifying data augmentations")


if __name__ == "__main__":
    main()
