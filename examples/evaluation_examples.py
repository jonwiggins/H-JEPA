#!/usr/bin/env python3
"""
Examples of using the H-JEPA evaluation framework.

This script demonstrates various ways to evaluate H-JEPA models
using the comprehensive evaluation framework.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.hjepa import create_hjepa
from src.data import build_dataset, build_dataloader
from src.evaluation import (
    linear_probe_eval,
    knn_eval,
    analyze_feature_quality,
    print_quality_report,
    fine_tune_eval,
    few_shot_eval,
    LinearProbeEvaluator,
    KNNEvaluator,
    FeatureQualityAnalyzer,
)


def example_1_linear_probe():
    """Example 1: Linear probe evaluation."""
    print("\n" + "="*80)
    print("Example 1: Linear Probe Evaluation")
    print("="*80)

    # Create model
    model = create_hjepa(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        embed_dim=768,
        num_hierarchies=3,
    )

    # Load checkpoint (example - replace with actual checkpoint)
    # checkpoint = torch.load('path/to/checkpoint.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Create datasets
    train_dataset = build_dataset("cifar10", "./data", split="train", download=True)
    val_dataset = build_dataset("cifar10", "./data", split="val", download=True)

    train_loader = build_dataloader(train_dataset, batch_size=256, num_workers=4)
    val_loader = build_dataloader(val_dataset, batch_size=256, num_workers=4)

    # Run linear probe evaluation
    print("\nRunning linear probe evaluation...")
    print("Note: This is just an example. Load a trained checkpoint for real evaluation.")

    # metrics = linear_probe_eval(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_classes=10,
    #     hierarchy_level=0,
    #     epochs=100,
    #     lr=0.1,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )
    #
    # print(f"\nResults:")
    # print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    # print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.2f}%")

    print("\nExample complete! Uncomment the code above to run actual evaluation.")


def example_2_knn_evaluation():
    """Example 2: k-NN evaluation."""
    print("\n" + "="*80)
    print("Example 2: k-NN Evaluation")
    print("="*80)

    # Create model
    model = create_hjepa()

    # Create datasets
    train_dataset = build_dataset("cifar10", "./data", split="train", download=True)
    test_dataset = build_dataset("cifar10", "./data", split="val", download=True)

    train_loader = build_dataloader(train_dataset, batch_size=256, num_workers=4, shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=256, num_workers=4, shuffle=False)

    print("\nRunning k-NN evaluation...")
    print("Note: Load a trained checkpoint for real evaluation.")

    # metrics = knn_eval(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     num_classes=10,
    #     hierarchy_level=0,
    #     k=20,
    #     distance_metric='cosine',
    #     temperature=0.07,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )
    #
    # print(f"\nResults:")
    # print(f"  k-NN Accuracy (k=20): {metrics['accuracy']:.2f}%")
    # print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.2f}%")

    print("\nExample complete!")


def example_3_feature_quality():
    """Example 3: Feature quality analysis."""
    print("\n" + "="*80)
    print("Example 3: Feature Quality Analysis")
    print("="*80)

    # Create model
    model = create_hjepa()

    # Create dataset
    dataset = build_dataset("cifar10", "./data", split="val", download=True)
    dataloader = build_dataloader(dataset, batch_size=256, num_workers=4)

    print("\nAnalyzing feature quality...")
    print("Note: Load a trained checkpoint for real analysis.")

    # metrics = analyze_feature_quality(
    #     model=model,
    #     dataloader=dataloader,
    #     hierarchy_level=0,
    #     max_samples=10000,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )
    #
    # # Print detailed report
    # print_quality_report(metrics, verbose=True)

    print("\nExample complete!")


def example_4_compare_hierarchies():
    """Example 4: Compare multiple hierarchy levels."""
    print("\n" + "="*80)
    print("Example 4: Compare Hierarchy Levels")
    print("="*80)

    # Create model
    model = create_hjepa(num_hierarchies=3)

    # Create datasets
    train_dataset = build_dataset("cifar10", "./data", split="train", download=True)
    val_dataset = build_dataset("cifar10", "./data", split="val", download=True)

    train_loader = build_dataloader(train_dataset, batch_size=256, num_workers=4)
    val_loader = build_dataloader(val_dataset, batch_size=256, num_workers=4)

    print("\nEvaluating all hierarchy levels...")

    results = {}

    for level in range(3):
        print(f"\n--- Hierarchy Level {level} ---")

        # You can run any evaluation for each level
        # Example: k-NN
        # metrics = knn_eval(
        #     model=model,
        #     train_loader=train_loader,
        #     test_loader=val_loader,
        #     num_classes=10,
        #     hierarchy_level=level,
        #     k=20,
        #     device='cuda' if torch.cuda.is_available() else 'cpu',
        # )
        # results[level] = metrics
        # print(f"  k-NN Accuracy: {metrics['accuracy']:.2f}%")

        print(f"  (Load checkpoint to run actual evaluation)")

    print("\nExample complete!")


def example_5_advanced_knn():
    """Example 5: Advanced k-NN with hyperparameter sweeping."""
    print("\n" + "="*80)
    print("Example 5: Advanced k-NN Hyperparameter Sweeping")
    print("="*80)

    # Create model and data
    model = create_hjepa()
    train_dataset = build_dataset("cifar10", "./data", split="train", download=True)
    test_dataset = build_dataset("cifar10", "./data", split="val", download=True)

    train_loader = build_dataloader(train_dataset, batch_size=256, num_workers=4, shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=256, num_workers=4, shuffle=False)

    # Create k-NN evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\nSweeping k-NN hyperparameters...")
    print("Note: Load a trained checkpoint for real evaluation.")

    # evaluator = KNNEvaluator(
    #     model=model,
    #     hierarchy_level=0,
    #     k=20,
    #     distance_metric='cosine',
    #     temperature=0.07,
    #     device=device,
    # )
    #
    # # Build k-NN index
    # evaluator.build_knn_index(train_loader)
    #
    # # Evaluate with different k values
    # k_values = [5, 10, 20, 50, 100]
    # results = evaluator.evaluate_multiple_k(
    #     test_loader=test_loader,
    #     num_classes=10,
    #     k_values=k_values,
    #     verbose=True,
    # )
    #
    # print("\nResults for different k values:")
    # for k, metrics in results.items():
    #     print(f"  k={k:3d}: {metrics['accuracy']:.2f}%")

    print("\nExample complete!")


def example_6_fine_tuning():
    """Example 6: Fine-tuning evaluation."""
    print("\n" + "="*80)
    print("Example 6: Fine-tuning Evaluation")
    print("="*80)

    # Create model
    model = create_hjepa()

    # Create datasets
    train_dataset = build_dataset("cifar10", "./data", split="train", download=True)
    val_dataset = build_dataset("cifar10", "./data", split="val", download=True)

    train_loader = build_dataloader(train_dataset, batch_size=128, num_workers=4)
    val_loader = build_dataloader(val_dataset, batch_size=128, num_workers=4)

    print("\nFine-tuning model...")
    print("Note: Load a pretrained checkpoint for real fine-tuning.")

    # Option 1: Frozen encoder (linear head only)
    # metrics_frozen = fine_tune_eval(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_classes=10,
    #     hierarchy_level=0,
    #     freeze_encoder=True,
    #     epochs=50,
    #     lr=1e-3,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )
    # print(f"\nFrozen Encoder: {metrics_frozen['accuracy']:.2f}%")

    # Option 2: Full fine-tuning
    # metrics_full = fine_tune_eval(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_classes=10,
    #     hierarchy_level=0,
    #     freeze_encoder=False,
    #     epochs=50,
    #     lr=1e-3,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )
    # print(f"Full Fine-tuning: {metrics_full['accuracy']:.2f}%")

    print("\nExample complete!")


def example_7_few_shot():
    """Example 7: Few-shot learning evaluation."""
    print("\n" + "="*80)
    print("Example 7: Few-shot Learning Evaluation")
    print("="*80)

    # Create model
    model = create_hjepa()

    # Create dataset
    dataset = build_dataset("cifar10", "./data", split="val", download=True)

    print("\nEvaluating few-shot learning...")
    print("Note: Load a trained checkpoint for real evaluation.")

    # results = few_shot_eval(
    #     model=model,
    #     dataset=dataset,
    #     num_classes=10,
    #     n_way=5,
    #     k_shot_list=[1, 5, 10],
    #     n_episodes=100,
    #     hierarchy_level=0,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )
    #
    # print("\nFew-shot results:")
    # for k_shot, metrics in results.items():
    #     print(f"  {k_shot}-shot: {metrics['accuracy']:.2f}% "
    #           f"± {metrics['confidence_interval']:.2f}%")

    print("\nExample complete!")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("H-JEPA Evaluation Framework - Examples")
    print("="*80)
    print("\nThese examples demonstrate how to use the evaluation framework.")
    print("Uncomment the evaluation code in each example to run actual evaluations.")
    print("\nMake sure to:")
    print("  1. Train a model or download a pretrained checkpoint")
    print("  2. Load the checkpoint before running evaluations")
    print("  3. Have the required datasets downloaded")

    # Run examples
    example_1_linear_probe()
    example_2_knn_evaluation()
    example_3_feature_quality()
    example_4_compare_hierarchies()
    example_5_advanced_knn()
    example_6_fine_tuning()
    example_7_few_shot()

    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80)
    print("\nTo run actual evaluations, use the main script:")
    print("  python scripts/evaluate.py --checkpoint model.pth --dataset cifar10")
    print("\nOr uncomment the code in the examples above and load a trained checkpoint.")


if __name__ == "__main__":
    main()
