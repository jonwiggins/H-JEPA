#!/usr/bin/env python3.11
"""
k-Nearest Neighbors Evaluation for H-JEPA

This script evaluates representation quality using k-NN classification.
Unlike linear probing, k-NN requires no training and directly measures
how well features cluster by semantic class.

This is a fast, training-free evaluation metric commonly used for
self-supervised learning models.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import get_dataset
from src.models.hjepa import create_hjepa


def load_pretrained_encoder(checkpoint_path: str, device: str) -> tuple:
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
    encoder = model.target_encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    encoder = encoder.to(device)

    print("✓ Encoder loaded and frozen")

    return encoder, config


def extract_features(
    encoder, dataloader: DataLoader, device: str, hierarchy_level: int = -1
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

            # Global average pooling
            features = F.adaptive_avg_pool2d(emb, 1).flatten(1)

            # L2 normalize for cosine similarity
            features = F.normalize(features, p=2, dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"✓ Extracted {len(features)} normalized feature vectors")

    return features, labels


def knn_classify(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
    batch_size: int = 256,
) -> tuple[float, np.ndarray]:
    """
    Perform k-NN classification with temperature-scaled similarities.

    Args:
        train_features: (N_train, D) normalized features
        train_labels: (N_train,) labels
        test_features: (N_test, D) normalized features
        test_labels: (N_test,) labels
        k: Number of nearest neighbors
        temperature: Temperature for softmax (lower = sharper)
        batch_size: Batch size for processing test samples

    Returns:
        accuracy: Top-1 accuracy
        predictions: Predicted labels for all test samples
    """
    num_classes = train_labels.max().item() + 1
    num_test = len(test_features)

    predictions = []
    correct = 0

    print(f"\nRunning k-NN classification (k={k})...")

    # Process test samples in batches to avoid OOM
    for i in tqdm(range(0, num_test, batch_size), desc="k-NN classification"):
        batch_features = test_features[i : i + batch_size]
        batch_labels = test_labels[i : i + batch_size]

        # Compute cosine similarities (features are already normalized)
        # Shape: (batch_size, N_train)
        similarities = batch_features @ train_features.T

        # Apply temperature scaling
        similarities = similarities / temperature

        # Get top-k nearest neighbors
        topk_similarities, topk_indices = similarities.topk(k, dim=1, largest=True)

        # Get labels of top-k neighbors
        # Shape: (batch_size, k)
        topk_labels = train_labels[topk_indices]

        # Weight votes by similarity (softmax over top-k)
        weights = F.softmax(topk_similarities, dim=1)

        # Compute weighted votes for each class
        batch_predictions = []
        for j in range(len(batch_features)):
            # Count weighted votes for each class
            class_votes = torch.zeros(num_classes)
            for c in range(num_classes):
                mask = topk_labels[j] == c
                class_votes[c] = weights[j][mask].sum()

            # Predict class with highest vote
            pred = class_votes.argmax().item()
            batch_predictions.append(pred)

            # Check if correct
            if pred == batch_labels[j].item():
                correct += 1

        predictions.extend(batch_predictions)

    accuracy = 100.0 * correct / num_test
    predictions = np.array(predictions)

    return accuracy, predictions


def compute_per_class_accuracy(
    predictions: np.ndarray, labels: torch.Tensor, num_classes: int
) -> dict:
    """Compute per-class accuracy"""
    labels_np = labels.numpy()
    per_class_acc = {}

    for c in range(num_classes):
        mask = labels_np == c
        if mask.sum() > 0:
            class_acc = (predictions[mask] == c).sum() / mask.sum() * 100
            per_class_acc[c] = class_acc

    return per_class_acc


def main():
    parser = argparse.ArgumentParser(description="k-NN evaluation for H-JEPA")
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
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20], help="k values to test")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for softmax")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/knn_eval", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("H-JEPA k-NN Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"k values: {args.k}")
    print(f"Temperature: {args.temperature}")
    print()

    # Load pretrained encoder
    encoder, config = load_pretrained_encoder(args.checkpoint, args.device)

    # Get number of classes
    num_classes = {"cifar10": 10, "cifar100": 100, "stl10": 10}[args.dataset]

    # Load datasets
    print(f"\nLoading {args.dataset} dataset...")

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = get_dataset(args.dataset, root=args.data_path, train=True, transform=transform)
    test_dataset = get_dataset(args.dataset, root=args.data_path, train=False, transform=transform)

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Extract features
    train_features, train_labels = extract_features(
        encoder, train_loader, args.device, args.hierarchy_level
    )
    test_features, test_labels = extract_features(
        encoder, test_loader, args.device, args.hierarchy_level
    )

    # Run k-NN for different k values
    results = {}

    for k in args.k:
        print(f"\n{'='*80}")
        print(f"Testing k={k}")
        print(f"{'='*80}")

        accuracy, predictions = knn_classify(
            train_features,
            train_labels,
            test_features,
            test_labels,
            k=k,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

        # Compute per-class accuracy
        per_class_acc = compute_per_class_accuracy(predictions, test_labels, num_classes)

        results[f"k={k}"] = {
            "accuracy": accuracy,
            "per_class_accuracy": {int(c): float(acc) for c, acc in per_class_acc.items()},
        }

        print(f"\n✓ k={k} Accuracy: {accuracy:.2f}%")
        print(f"  Per-class accuracy (mean): {np.mean(list(per_class_acc.values())):.2f}%")

    # Save results
    checkpoint_name = Path(args.checkpoint).stem
    results_file = output_dir / f"{checkpoint_name}_{args.dataset}_knn_results.json"

    results_summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "hierarchy_level": args.hierarchy_level,
        "feature_dim": train_features.shape[1],
        "num_classes": num_classes,
        "temperature": args.temperature,
        "results": results,
        "config": config,
    }

    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("k-NN EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Feature dimension: {train_features.shape[1]}")
    print()
    for k in args.k:
        acc = results[f"k={k}"]["accuracy"]
        print(f"  k={k:2d}: {acc:6.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
