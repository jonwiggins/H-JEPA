#!/usr/bin/env python3.11
"""
Linear Probing Evaluation for H-JEPA

This script evaluates the quality of learned representations by freezing
the pretrained encoder and training a linear classifier on top.

This is the gold standard for evaluating self-supervised learning models.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import get_dataset
from src.models.hjepa import create_hjepa


class LinearClassifier(nn.Module):
    """Simple linear classifier head"""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def load_pretrained_encoder(checkpoint_path: str, device: str) -> tuple[nn.Module, dict]:
    """Load pretrained H-JEPA encoder"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", {})
    model_state = checkpoint.get("model_state_dict", checkpoint.get("target_encoder", {}))

    # Create full H-JEPA model
    model = create_hjepa(
        encoder_type=config.get("model", {}).get("encoder_type", "vit_base_patch16_224"),
        img_size=config.get("data", {}).get("image_size", 224),
        num_hierarchies=config.get("model", {}).get("num_hierarchies", 3),
        predictor_depth=config.get("model", {}).get("predictor", {}).get("depth", 6),
        predictor_heads=config.get("model", {}).get("predictor", {}).get("num_heads", 6),
        use_rope=config.get("model", {}).get("use_rope", True),
        use_flash_attention=config.get("model", {}).get("use_flash_attention", True),
    )

    # Load weights
    model.load_state_dict(model_state, strict=False)

    # Extract and freeze the encoder
    encoder = model.target_encoder  # Use target encoder (EMA weights)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    encoder = encoder.to(device)

    print("✓ Encoder loaded and frozen")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    return encoder, config


def extract_features(
    encoder: nn.Module, dataloader: DataLoader, device: str, hierarchy_level: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from the encoder"""
    all_features = []
    all_labels = []

    print(f"Extracting features (hierarchy level: {hierarchy_level})...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Feature extraction"):
            images = images.to(device)

            # Get hierarchical embeddings
            embeddings = encoder.forward_hierarchical(images)

            # Select hierarchy level (-1 = highest level)
            emb = embeddings[hierarchy_level]

            # Global average pooling to get single vector per image
            features = F.adaptive_avg_pool2d(emb, 1).flatten(1)

            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"✓ Extracted {len(features)} feature vectors of dimension {features.shape[1]}")

    return features, labels


def train_linear_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    device: str,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 256,
) -> dict:
    """Train linear classifier on frozen features"""

    # Create classifier
    input_dim = train_features.shape[1]
    classifier = LinearClassifier(input_dim, num_classes).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Create dataloaders for features
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_acc = 0.0
    results = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    print("\nTraining linear classifier...")
    print(f"  Input dim: {input_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")

    for epoch in range(epochs):
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                logits = classifier(features)
                loss = F.cross_entropy(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # Update best accuracy
        if val_acc > best_acc:
            best_acc = val_acc

        # Store results
        results["train_acc"].append(train_acc)
        results["val_acc"].append(val_acc)
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

        scheduler.step()

    results["best_val_acc"] = best_acc
    results["final_val_acc"] = val_acc

    print("\n✓ Training complete!")
    print(f"  Best validation accuracy: {best_acc:.2f}%")
    print(f"  Final validation accuracy: {val_acc:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Linear probing evaluation for H-JEPA")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to pretrained checkpoint"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10"]
    )
    parser.add_argument("--data-path", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument(
        "--hierarchy-level", type=int, default=-1, help="Which hierarchy to use (-1 = highest)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs for linear classifier"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--output-dir", type=str, default="results/linear_probe", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("H-JEPA Linear Probing Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Hierarchy level: {args.hierarchy_level}")
    print()

    # Load pretrained encoder
    encoder, config = load_pretrained_encoder(args.checkpoint, args.device)

    # Get number of classes
    num_classes = {"cifar10": 10, "cifar100": 100, "stl10": 10}[args.dataset]

    # Load datasets
    print(f"\nLoading {args.dataset} dataset...")

    # Standard normalization for evaluation
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = get_dataset(args.dataset, root=args.data_path, train=True, transform=transform)
    val_dataset = get_dataset(args.dataset, root=args.data_path, train=False, transform=transform)

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")

    # Create dataloaders for feature extraction
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Extract features
    train_features, train_labels = extract_features(
        encoder, train_loader, args.device, args.hierarchy_level
    )
    val_features, val_labels = extract_features(
        encoder, val_loader, args.device, args.hierarchy_level
    )

    # Train linear classifier
    results = train_linear_classifier(
        train_features,
        train_labels,
        val_features,
        val_labels,
        num_classes,
        args.device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # Save results
    checkpoint_name = Path(args.checkpoint).stem
    results_file = output_dir / f"{checkpoint_name}_{args.dataset}_results.json"

    results_summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "hierarchy_level": args.hierarchy_level,
        "feature_dim": train_features.shape[1],
        "num_classes": num_classes,
        "best_val_acc": results["best_val_acc"],
        "final_val_acc": results["final_val_acc"],
        "config": config,
    }

    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"Final validation accuracy: {results['final_val_acc']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
