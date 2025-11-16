#!/usr/bin/env python3.11
"""
Transfer Learning Evaluation for H-JEPA

This script evaluates transfer learning performance by testing pretrained
representations on multiple downstream tasks. It runs both linear probing
and k-NN evaluation across different datasets.

This provides a comprehensive assessment of representation generalization.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

DATASETS = ["cifar10", "cifar100", "stl10"]


def run_linear_probe(
    checkpoint: str,
    dataset: str,
    device: str,
    epochs: int,
    output_dir: Path,
) -> Dict:
    """Run linear probing evaluation"""
    print(f"\n{'='*80}")
    print(f"Running Linear Probing on {dataset.upper()}")
    print(f"{'='*80}")

    cmd = [
        "python3.11",
        "scripts/eval_linear_probe.py",
        "--checkpoint",
        checkpoint,
        "--dataset",
        dataset,
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--output-dir",
        str(output_dir / "linear_probe"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Linear probing failed for {dataset}")
        print(result.stderr)
        return {"error": result.stderr}

    # Load results
    checkpoint_name = Path(checkpoint).stem
    results_file = output_dir / "linear_probe" / f"{checkpoint_name}_{dataset}_results.json"

    with open(results_file, "r") as f:
        results = json.load(f)

    print(f"✓ Linear Probe Accuracy: {results['best_val_acc']:.2f}%")

    return results


def run_knn(
    checkpoint: str,
    dataset: str,
    device: str,
    k_values: List[int],
    output_dir: Path,
) -> Dict:
    """Run k-NN evaluation"""
    print(f"\n{'='*80}")
    print(f"Running k-NN on {dataset.upper()}")
    print(f"{'='*80}")

    k_str = " ".join(str(k) for k in k_values)

    cmd = [
        "python3.11",
        "scripts/eval_knn.py",
        "--checkpoint",
        checkpoint,
        "--dataset",
        dataset,
        "--device",
        device,
        "--k",
        *[str(k) for k in k_values],
        "--output-dir",
        str(output_dir / "knn"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ k-NN failed for {dataset}")
        print(result.stderr)
        return {"error": result.stderr}

    # Load results
    checkpoint_name = Path(checkpoint).stem
    results_file = output_dir / "knn" / f"{checkpoint_name}_{dataset}_knn_results.json"

    with open(results_file, "r") as f:
        results = json.load(f)

    # Print k-NN results
    print(f"✓ k-NN Results:")
    for k in k_values:
        acc = results["results"][f"k={k}"]["accuracy"]
        print(f"  k={k:2d}: {acc:6.2f}%")

    return results


def create_summary_report(
    linear_results: Dict[str, Dict],
    knn_results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """Create a summary report of all results"""

    # Prepare data for table
    data = []

    for dataset in DATASETS:
        if dataset in linear_results and "best_val_acc" in linear_results[dataset]:
            row = {
                "Dataset": dataset.upper(),
                "Linear Probe": f"{linear_results[dataset]['best_val_acc']:.2f}%",
            }

            # Add k-NN results
            if dataset in knn_results and "results" in knn_results[dataset]:
                for k_key, k_result in knn_results[dataset]["results"].items():
                    row[f"k-NN {k_key}"] = f"{k_result['accuracy']:.2f}%"

            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Print table
    print("\n" + "=" * 80)
    print("TRANSFER LEARNING EVALUATION SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Save to CSV
    csv_file = output_dir / "transfer_learning_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Summary saved to {csv_file}")

    # Save detailed JSON
    json_file = output_dir / "transfer_learning_detailed.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "linear_probe": linear_results,
                "knn": knn_results,
            },
            f,
            indent=2,
        )

    print(f"✓ Detailed results saved to {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Transfer learning evaluation for H-JEPA")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to pretrained checkpoint"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Datasets to evaluate on",
    )
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--linear-probe", action="store_true", help="Run linear probing evaluation")
    parser.add_argument("--knn", action="store_true", help="Run k-NN evaluation")
    parser.add_argument("--linear-epochs", type=int, default=100, help="Epochs for linear probing")
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 5, 10, 20], help="k values for k-NN"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/transfer_learning", help="Output directory"
    )

    args = parser.parse_args()

    # If neither specified, run both
    if not args.linear_probe and not args.knn:
        args.linear_probe = True
        args.knn = True

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("H-JEPA Transfer Learning Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Device: {args.device}")
    print(f"Evaluations: ", end="")
    if args.linear_probe:
        print("Linear Probe ", end="")
    if args.knn:
        print("k-NN", end="")
    print("\n" + "=" * 80)

    linear_results = {}
    knn_results = {}

    # Run evaluations on each dataset
    for dataset in args.datasets:
        print(f"\n{'#'*80}")
        print(f"# Evaluating on {dataset.upper()}")
        print(f"{'#'*80}")

        if args.linear_probe:
            linear_results[dataset] = run_linear_probe(
                args.checkpoint,
                dataset,
                args.device,
                args.linear_epochs,
                output_dir,
            )

        if args.knn:
            knn_results[dataset] = run_knn(
                args.checkpoint,
                dataset,
                args.device,
                args.k_values,
                output_dir,
            )

    # Create summary report
    create_summary_report(linear_results, knn_results, output_dir)

    print("\n" + "=" * 80)
    print("✓ Transfer learning evaluation complete!")
    print(f"✓ Results saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
