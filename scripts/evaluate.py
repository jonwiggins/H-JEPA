#!/usr/bin/env python3
"""
Comprehensive evaluation script for H-JEPA.

This script runs various evaluation protocols on a pretrained H-JEPA model:
- Linear probe evaluation
- k-NN classification
- Feature quality analysis
- Transfer learning (fine-tuning)
- Few-shot learning

Usage:
    # Run all evaluations
    python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth --dataset cifar10

    # Run specific evaluation
    python scripts/evaluate.py --checkpoint model.pth --dataset cifar10 --eval-type linear_probe

    # Multiple hierarchy levels
    python scripts/evaluate.py --checkpoint model.pth --dataset cifar10 --hierarchy-levels 0 1 2
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data import build_dataloader, build_dataset
from src.evaluation.feature_quality import (
    FeatureQualityAnalyzer,
    analyze_feature_quality,
    compare_hierarchy_levels,
    print_quality_report,
)
from src.evaluation.knn_eval import KNNEvaluator, knn_eval
from src.evaluation.linear_probe import LinearProbeEvaluator, linear_probe_eval
from src.evaluation.transfer import few_shot_eval, fine_tune_eval
from src.models.hjepa import create_hjepa


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of H-JEPA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100", "imagenet", "imagenet100", "stl10"],
        help="Dataset to evaluate on",
    )

    # Evaluation type
    parser.add_argument(
        "--eval-type",
        type=str,
        default="all",
        choices=["all", "linear_probe", "knn", "feature_quality", "fine_tune", "few_shot"],
        help="Type of evaluation to perform",
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to dataset directory",
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
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )

    # Model arguments
    parser.add_argument(
        "--hierarchy-levels",
        type=int,
        nargs="+",
        default=[0],
        help="Hierarchy levels to evaluate (e.g., 0 1 2)",
    )

    # Linear probe arguments
    parser.add_argument(
        "--linear-probe-epochs",
        type=int,
        default=100,
        help="Number of epochs for linear probe training",
    )
    parser.add_argument(
        "--linear-probe-lr",
        type=float,
        default=0.1,
        help="Learning rate for linear probe",
    )

    # k-NN arguments
    parser.add_argument(
        "--knn-k",
        type=int,
        default=20,
        help="k for k-NN evaluation",
    )
    parser.add_argument(
        "--knn-temperature",
        type=float,
        default=0.07,
        help="Temperature for k-NN distance weighting",
    )

    # Fine-tuning arguments
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=50,
        help="Number of epochs for fine-tuning",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-3,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder during fine-tuning",
    )

    # Few-shot arguments
    parser.add_argument(
        "--few-shot-n-way",
        type=int,
        default=5,
        help="Number of classes per few-shot episode",
    )
    parser.add_argument(
        "--few-shot-k-shots",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Number of examples per class for few-shot",
    )
    parser.add_argument(
        "--few-shot-episodes",
        type=int,
        default=100,
        help="Number of few-shot episodes to evaluate",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save feature visualizations (t-SNE, UMAP)",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


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


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load H-JEPA model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"\nLoading checkpoint from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Print checkpoint info
    print(f"Checkpoint info:")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "best_loss" in checkpoint:
        print(f"  Best loss: {checkpoint['best_loss']:.4f}")

    # Create model from checkpoint config or use defaults
    if "config" in checkpoint:
        config = checkpoint["config"]
        model = create_hjepa(
            encoder_type=config.get("model", {}).get("encoder_type", "vit_base_patch16_224"),
            img_size=config.get("data", {}).get("image_size", 224),
            embed_dim=config.get("model", {}).get("embed_dim", 768),
            num_hierarchies=config.get("model", {}).get("num_hierarchies", 3),
        )
    else:
        # Use default configuration
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

    print(f"Model loaded successfully")
    print(f"  Embedding dimension: {model.embed_dim}")
    print(f"  Number of hierarchies: {model.num_hierarchies}")

    return model


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for dataset."""
    class_map = {
        "cifar10": 10,
        "stl10": 10,
        "cifar100": 100,
        "imagenet100": 100,
        "imagenet": 1000,
    }
    return class_map[dataset_name]


def run_linear_probe(model, train_loader, val_loader, num_classes, hierarchy_level, args, device):
    """Run linear probe evaluation."""
    print("\n" + "=" * 80)
    print(f"Linear Probe Evaluation - Hierarchy Level {hierarchy_level}")
    print("=" * 80)

    metrics = linear_probe_eval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        hierarchy_level=hierarchy_level,
        epochs=args.linear_probe_epochs,
        lr=args.linear_probe_lr,
        device=str(device),
        verbose=args.verbose,
    )

    print(f"\nLinear Probe Results (Level {hierarchy_level}):")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.2f}%")

    return metrics


def run_knn(model, train_loader, test_loader, num_classes, hierarchy_level, args, device):
    """Run k-NN evaluation."""
    print("\n" + "=" * 80)
    print(f"k-NN Evaluation - Hierarchy Level {hierarchy_level}")
    print("=" * 80)

    metrics = knn_eval(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        hierarchy_level=hierarchy_level,
        k=args.knn_k,
        temperature=args.knn_temperature,
        device=str(device),
        verbose=args.verbose,
    )

    print(f"\nk-NN Results (Level {hierarchy_level}):")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")

    return metrics


def run_feature_quality(model, dataloader, hierarchy_level, args, device):
    """Run feature quality analysis."""
    print("\n" + "=" * 80)
    print(f"Feature Quality Analysis - Hierarchy Level {hierarchy_level}")
    print("=" * 80)

    metrics = analyze_feature_quality(
        model=model,
        dataloader=dataloader,
        hierarchy_level=hierarchy_level,
        max_samples=10000,
        device=str(device),
    )

    print_quality_report(metrics, verbose=args.verbose)

    return metrics


def run_fine_tune(model, train_loader, val_loader, num_classes, hierarchy_level, args, device):
    """Run fine-tuning evaluation."""
    print("\n" + "=" * 80)
    print(f"Fine-tuning Evaluation - Hierarchy Level {hierarchy_level}")
    print("=" * 80)

    metrics = fine_tune_eval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        hierarchy_level=hierarchy_level,
        freeze_encoder=args.freeze_encoder,
        epochs=args.fine_tune_epochs,
        lr=args.fine_tune_lr,
        device=str(device),
        verbose=args.verbose,
    )

    mode = "frozen" if args.freeze_encoder else "full"
    print(f"\nFine-tuning Results ({mode}, Level {hierarchy_level}):")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")

    return metrics


def run_few_shot(model, dataset, num_classes, hierarchy_level, args, device):
    """Run few-shot evaluation."""
    print("\n" + "=" * 80)
    print(f"Few-shot Learning Evaluation - Hierarchy Level {hierarchy_level}")
    print("=" * 80)

    results = few_shot_eval(
        model=model,
        dataset=dataset,
        num_classes=num_classes,
        n_way=args.few_shot_n_way,
        k_shot_list=args.few_shot_k_shots,
        n_episodes=args.few_shot_episodes,
        hierarchy_level=hierarchy_level,
        device=str(device),
        verbose=args.verbose,
    )

    print(f"\nFew-shot Results (Level {hierarchy_level}):")
    for k_shot, metrics in results.items():
        print(
            f"  {k_shot}-shot: {metrics['accuracy']:.2f}% ± {metrics['confidence_interval']:.2f}%"
        )

    return results


def save_results(results: dict, output_dir: str, args):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Add metadata
    results["metadata"] = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "eval_type": args.eval_type,
        "hierarchy_levels": args.hierarchy_levels,
    }

    # Save to JSON
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = setup_device(args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("H-JEPA Evaluation")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Evaluation type: {args.eval_type}")
    print(f"Hierarchy levels: {args.hierarchy_levels}")
    print(f"Output directory: {args.output_dir}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Get number of classes
    num_classes = get_num_classes(args.dataset)

    # Build datasets
    print("\nLoading datasets...")
    train_dataset = build_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split="train",
        image_size=args.image_size,
        download=True,
    )

    val_dataset = build_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split="val",
        image_size=args.image_size,
        download=True,
    )

    # Build dataloaders
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # No need to shuffle for evaluation
        drop_last=False,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # Results storage
    all_results = {}

    # Run evaluations for each hierarchy level
    for level in args.hierarchy_levels:
        if level >= model.num_hierarchies:
            print(
                f"\nWarning: Skipping level {level} (model only has {model.num_hierarchies} levels)"
            )
            continue

        level_results = {}

        # Linear probe
        if args.eval_type in ["all", "linear_probe"]:
            level_results["linear_probe"] = run_linear_probe(
                model, train_loader, val_loader, num_classes, level, args, device
            )

        # k-NN
        if args.eval_type in ["all", "knn"]:
            level_results["knn"] = run_knn(
                model, train_loader, val_loader, num_classes, level, args, device
            )

        # Feature quality
        if args.eval_type in ["all", "feature_quality"]:
            level_results["feature_quality"] = run_feature_quality(
                model, val_loader, level, args, device
            )

        # Fine-tuning
        if args.eval_type in ["all", "fine_tune"]:
            level_results["fine_tune"] = run_fine_tune(
                model, train_loader, val_loader, num_classes, level, args, device
            )

        # Few-shot
        if args.eval_type in ["all", "few_shot"]:
            level_results["few_shot"] = run_few_shot(
                model, val_dataset, num_classes, level, args, device
            )

        all_results[f"level_{level}"] = level_results

    # Save results
    save_results(all_results, args.output_dir, args)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for level in args.hierarchy_levels:
        if f"level_{level}" not in all_results:
            continue

        print(f"\nHierarchy Level {level}:")
        level_results = all_results[f"level_{level}"]

        if "linear_probe" in level_results:
            print(f"  Linear Probe: {level_results['linear_probe']['accuracy']:.2f}%")

        if "knn" in level_results:
            print(f"  k-NN: {level_results['knn']['accuracy']:.2f}%")

        if "fine_tune" in level_results:
            print(f"  Fine-tune: {level_results['fine_tune']['accuracy']:.2f}%")

        if "feature_quality" in level_results:
            rank = level_results["feature_quality"]["rank"]
            print(f"  Effective Rank: {rank['effective_rank']:.2f}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
