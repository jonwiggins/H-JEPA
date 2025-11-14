#!/usr/bin/env python3
"""
Quick evaluation script for H-JEPA.

Simplified version of evaluate.py for rapid testing and validation.
Useful for quick sanity checks without running full evaluation suite.

Usage:
    # Quick k-NN evaluation (no training required)
    python scripts/quick_eval.py --checkpoint model.pth --method knn

    # Quick linear probe (fast, 20 epochs)
    python scripts/quick_eval.py --checkpoint model.pth --method linear_probe

    # Feature quality check
    python scripts/quick_eval.py --checkpoint model.pth --method feature_quality
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from src.models.hjepa import create_hjepa
from src.data import build_dataset, build_dataloader
from src.evaluation.linear_probe import linear_probe_eval
from src.evaluation.knn_eval import knn_eval
from src.evaluation.feature_quality import analyze_feature_quality, print_quality_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick evaluation of H-JEPA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="knn",
        choices=["knn", "linear_probe", "feature_quality", "all"],
        help="Evaluation method (knn is fastest)",
    )

    # Optional arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "stl10"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Hierarchy level to evaluate (0=finest)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--linear-probe-epochs",
        type=int,
        default=20,
        help="Number of epochs for linear probe (reduced for quick eval)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/quick_eval",
        help="Directory to save results",
    )

    return parser.parse_args()


def setup_device(args):
    """Setup device for evaluation."""
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*80}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "best_loss" in checkpoint:
        print(f"  Best loss: {checkpoint['best_loss']:.4f}")

    # Create model from checkpoint config or use defaults
    if "config" in checkpoint:
        config = checkpoint["config"]
        model = create_hjepa(
            encoder_type=config.get("model", {}).get("encoder_type", "vit_tiny_patch16_224"),
            img_size=config.get("data", {}).get("image_size", 224),
            embed_dim=config.get("model", {}).get("embed_dim", 192),
            num_hierarchies=config.get("model", {}).get("num_hierarchies", 2),
        )
    else:
        print("Warning: No config in checkpoint, using default model configuration")
        model = create_hjepa()

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"  Model loaded successfully")
    print(f"  Embedding dimension: {model.embed_dim}")
    print(f"  Number of hierarchies: {model.num_hierarchies}")

    return model


def get_num_classes(dataset_name):
    """Get number of classes for dataset."""
    class_map = {
        "cifar10": 10,
        "stl10": 10,
        "cifar100": 100,
    }
    return class_map[dataset_name]


def run_knn_eval(model, train_loader, test_loader, num_classes, level, device):
    """Run quick k-NN evaluation."""
    print(f"\n{'='*80}")
    print("k-NN Evaluation (No Training Required)")
    print(f"{'='*80}")

    start_time = datetime.now()

    metrics = knn_eval(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        hierarchy_level=level,
        k=20,
        temperature=0.07,
        device=str(device),
        verbose=True,
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print("k-NN Results")
    print(f"{'='*80}")
    print(f"  Hierarchy Level: {level}")
    print(f"  k: 20")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Evaluation Time: {elapsed:.1f} seconds")
    print(f"{'='*80}")

    return metrics


def run_linear_probe_eval(model, train_loader, val_loader, num_classes, level, epochs, device):
    """Run quick linear probe evaluation."""
    print(f"\n{'='*80}")
    print(f"Linear Probe Evaluation ({epochs} epochs)")
    print(f"{'='*80}")

    start_time = datetime.now()

    metrics = linear_probe_eval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        hierarchy_level=level,
        epochs=epochs,
        lr=0.1,
        device=str(device),
        verbose=True,
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print("Linear Probe Results")
    print(f"{'='*80}")
    print(f"  Hierarchy Level: {level}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.2f}%")
    print(f"  Training Time: {elapsed/60:.1f} minutes")
    print(f"{'='*80}")

    return metrics


def run_feature_quality_eval(model, dataloader, level, device):
    """Run feature quality analysis."""
    print(f"\n{'='*80}")
    print("Feature Quality Analysis")
    print(f"{'='*80}")

    start_time = datetime.now()

    metrics = analyze_feature_quality(
        model=model,
        dataloader=dataloader,
        hierarchy_level=level,
        max_samples=5000,  # Reduced for quick eval
        device=str(device),
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print("Feature Quality Results")
    print(f"{'='*80}")
    print_quality_report(metrics, verbose=True)
    print(f"\nAnalysis Time: {elapsed:.1f} seconds")
    print(f"{'='*80}")

    return metrics


def save_results(results, output_dir, args):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Add metadata
    results["metadata"] = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "method": args.method,
        "hierarchy_level": args.level,
        "timestamp": datetime.now().isoformat(),
    }

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"quick_eval_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print("H-JEPA Quick Evaluation")
    print(f"{'='*80}")
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Hierarchy Level: {args.level}")
    print(f"{'='*80}")

    # Setup device
    device = setup_device(args)

    # Load model
    model = load_checkpoint(args.checkpoint, device)

    # Check if requested level is valid
    if args.level >= model.num_hierarchies:
        print(f"\nWarning: Requested level {args.level} but model only has {model.num_hierarchies} levels")
        print(f"Using level 0 instead")
        args.level = 0

    # Get number of classes
    num_classes = get_num_classes(args.dataset)

    # Build datasets
    print(f"\n{'='*80}")
    print("Loading datasets...")
    print(f"{'='*80}")

    train_dataset = build_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split="train",
        image_size=224,
        download=True,
    )

    val_dataset = build_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split="val",
        image_size=224,
        download=True,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Build dataloaders
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # Run evaluation
    results = {}

    if args.method in ["knn", "all"]:
        results["knn"] = run_knn_eval(
            model, train_loader, val_loader, num_classes, args.level, device
        )

    if args.method in ["linear_probe", "all"]:
        results["linear_probe"] = run_linear_probe_eval(
            model, train_loader, val_loader, num_classes, args.level, args.linear_probe_epochs, device
        )

    if args.method in ["feature_quality", "all"]:
        results["feature_quality"] = run_feature_quality_eval(
            model, val_loader, args.level, device
        )

    # Save results if requested
    if args.save_results:
        save_results(results, args.output_dir, args)

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")

    if "knn" in results:
        print(f"k-NN Accuracy: {results['knn']['accuracy']:.2f}%")

    if "linear_probe" in results:
        print(f"Linear Probe Accuracy: {results['linear_probe']['accuracy']:.2f}%")

    if "feature_quality" in results:
        effective_rank = results["feature_quality"]["rank"]["effective_rank"]
        rank_ratio = results["feature_quality"]["rank"]["rank_ratio"]
        print(f"Effective Rank: {effective_rank:.1f} ({rank_ratio:.1%} of dimensions)")

    print(f"{'='*80}")

    # Provide interpretation
    print("\nInterpretation:")

    if "linear_probe" in results:
        acc = results["linear_probe"]["accuracy"]
        if acc >= 75:
            print("  ✅ Excellent: Model learned strong discriminative features")
        elif acc >= 70:
            print("  ✅ Good: Model learned useful representations")
        elif acc >= 60:
            print("  ⚠️  Moderate: Model learned basic features, room for improvement")
        else:
            print("  ❌ Poor: Model did not learn well, check training")

    if "feature_quality" in results:
        rank_ratio = results["feature_quality"]["rank"]["rank_ratio"]
        if rank_ratio >= 0.45:
            print("  ✅ No collapse: Features use many dimensions")
        elif rank_ratio >= 0.30:
            print("  ⚠️  Moderate rank: Some dimension collapse")
        else:
            print("  ❌ Severe collapse: Most dimensions unused")

    print()


if __name__ == "__main__":
    main()
