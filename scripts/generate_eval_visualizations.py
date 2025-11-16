#!/usr/bin/env python3
"""
Generate evaluation visualizations for H-JEPA.

This script creates comprehensive visualizations for evaluation results,
including confusion matrices, t-SNE plots, rank analysis, and comparisons.

Usage:
    python scripts/generate_eval_visualizations.py \
        --results results/evaluation/evaluation_results.json \
        --output-dir results/evaluation/visualizations
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150


def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/evaluation/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots",
    )
    return parser.parse_args()


def load_results(results_path):
    """Load evaluation results from JSON."""
    with open(results_path, "r") as f:
        results = json.load(f)
    return results


def plot_hierarchy_comparison(results, output_dir, fmt="png"):
    """Plot performance comparison across hierarchy levels."""
    print("Generating hierarchy comparison plot...")

    if "hierarchy_comparison" not in results:
        print("  Skipping - no hierarchy comparison data")
        return

    comp = results["hierarchy_comparison"]

    # Extract data
    metrics = []
    level_0_values = []
    level_1_values = []

    for metric_name, metric_data in comp.items():
        if isinstance(metric_data, dict) and "level_0" in metric_data:
            metrics.append(metric_name.replace("_", " ").title())
            level_0_values.append(metric_data["level_0"])
            level_1_values.append(metric_data.get("level_1", 0))

    # Create plot
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, level_0_values, width, label="Level 0 (Fine)", color="steelblue")
    bars2 = ax.bar(x + width / 2, level_1_values, width, label="Level 1 (Coarse)", color="coral")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Performance Across Hierarchy Levels", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"hierarchy_comparison.{fmt}")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved to {output_path}")


def plot_knn_sweep(results, output_dir, fmt="png"):
    """Plot k-NN hyperparameter sweep results."""
    print("Generating k-NN sweep plot...")

    # Find k-NN results
    knn_results = None
    for level_key, level_data in results.items():
        if level_key.startswith("level_") and "knn" in level_data:
            knn_results = level_data["knn"]
            break

    if not knn_results or "k_sweep_results" not in knn_results:
        print("  Skipping - no k-NN sweep data")
        return

    sweep = knn_results["k_sweep_results"]

    # Extract k values and accuracies
    k_values = []
    accuracies = []
    top5_accuracies = []

    for k_name, k_data in sweep.items():
        if k_name.startswith("k_"):
            k = int(k_name.split("_")[1])
            k_values.append(k)
            if isinstance(k_data, dict):
                accuracies.append(k_data.get("accuracy", k_data))
                top5_accuracies.append(k_data.get("top_5_accuracy", 0))
            else:
                accuracies.append(k_data)

    # Sort by k
    sorted_data = sorted(zip(k_values, accuracies, top5_accuracies))
    k_values = [x[0] for x in sorted_data]
    accuracies = [x[1] for x in sorted_data]
    top5_accuracies = [x[2] for x in sorted_data]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        k_values,
        accuracies,
        "o-",
        linewidth=2,
        markersize=8,
        label="Top-1 Accuracy",
        color="steelblue",
    )

    if any(top5_accuracies):
        ax.plot(
            k_values,
            top5_accuracies,
            "s--",
            linewidth=2,
            markersize=8,
            label="Top-5 Accuracy",
            color="coral",
            alpha=0.7,
        )

    # Mark optimal k
    optimal_k = knn_results.get("optimal_k", k_values[np.argmax(accuracies)])
    optimal_acc = max(accuracies)
    ax.axvline(x=optimal_k, color="red", linestyle="--", alpha=0.5, label=f"Optimal k={optimal_k}")
    ax.plot(optimal_k, optimal_acc, "r*", markersize=15)

    ax.set_xlabel("k (Number of Neighbors)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("k-NN Performance vs k", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"knn_k_sweep.{fmt}")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved to {output_path}")


def plot_per_class_accuracy(results, output_dir, fmt="png"):
    """Plot per-class accuracy bar chart."""
    print("Generating per-class accuracy plot...")

    # Find linear probe results with per-class accuracy
    linear_probe = None
    for level_key, level_data in results.items():
        if level_key.startswith("level_") and "linear_probe" in level_data:
            if "per_class_accuracy" in level_data["linear_probe"]:
                linear_probe = level_data["linear_probe"]
                break

    if not linear_probe:
        print("  Skipping - no per-class accuracy data")
        return

    per_class = linear_probe["per_class_accuracy"]

    # Sort by accuracy
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    classes = [x[0].capitalize() for x in sorted_classes]
    accuracies = [x[1] for x in sorted_classes]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        "darkgreen" if acc >= 80 else "steelblue" if acc >= 70 else "coral" for acc in accuracies
    ]

    bars = ax.barh(classes, accuracies, color=colors, alpha=0.8)

    # Add accuracy labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, i, f"{acc:.1f}%", va="center", fontsize=10)

    # Add vertical lines for thresholds
    ax.axvline(x=80, color="green", linestyle="--", alpha=0.5, label="Excellent (≥80%)")
    ax.axvline(x=70, color="blue", linestyle="--", alpha=0.5, label="Good (≥70%)")

    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.set_title("Per-Class Linear Probe Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"per_class_accuracy.{fmt}")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved to {output_path}")


def plot_feature_quality_summary(results, output_dir, fmt="png"):
    """Plot feature quality metrics summary."""
    print("Generating feature quality summary...")

    # Find feature quality results
    feature_quality = None
    for level_key, level_data in results.items():
        if level_key.startswith("level_") and "feature_quality" in level_data:
            feature_quality = level_data["feature_quality"]
            break

    if not feature_quality:
        print("  Skipping - no feature quality data")
        return

    # Extract metrics
    rank = feature_quality.get("rank", {})
    stats = feature_quality.get("statistics", {})
    isotropy = feature_quality.get("isotropy", {})

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rank metrics
    ax = axes[0, 0]
    metrics = ["Effective\nRank", "Rank\nRatio", "Components\nfor 95%", "Components\nfor 99%"]
    values = [
        rank.get("effective_rank", 0),
        rank.get("rank_ratio", 0) * 100,  # Convert to percentage
        rank.get("num_components_95", 0),
        rank.get("num_components_99", 0),
    ]
    colors = ["steelblue", "coral", "lightgreen", "gold"]
    bars = ax.bar(metrics, values, color=colors, alpha=0.8)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Rank Analysis", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 2. Singular values (if available)
    ax = axes[0, 1]
    if "singular_values_top_10" in rank:
        sv = rank["singular_values_top_10"]
        ax.plot(range(1, len(sv) + 1), sv, "o-", linewidth=2, markersize=8, color="steelblue")
        ax.set_xlabel("Component Index", fontsize=11)
        ax.set_ylabel("Singular Value", fontsize=11)
        ax.set_title("Top 10 Singular Values", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No singular value data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")

    # 3. Variance statistics
    ax = axes[1, 0]
    stat_names = ["Mean\nVariance", "Std\nVariance", "Coeff. of\nVariation"]
    stat_values = [
        stats.get("mean_variance", 0),
        stats.get("std_variance", 0),
        stats.get("coefficient_variation", 0),
    ]
    bars = ax.bar(stat_names, stat_values, color="coral", alpha=0.8)

    for bar, val in zip(bars, stat_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Variance Statistics", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 4. Isotropy metrics
    ax = axes[1, 1]
    iso_metrics = ["Uniformity", "Mean Cosine\nSimilarity"]
    iso_values = [isotropy.get("uniformity", 0), isotropy.get("mean_cosine_similarity", 0)]
    colors = [
        "lightgreen" if isotropy.get("uniformity", 0) < -2.0 else "coral",
        "lightgreen" if isotropy.get("mean_cosine_similarity", 1) < 0.3 else "coral",
    ]
    bars = ax.bar(iso_metrics, iso_values, color=colors, alpha=0.8)

    for bar, val in zip(bars, iso_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom" if val > 0 else "top",
            fontsize=10,
        )

    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Isotropy Metrics", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(alpha=0.3)

    plt.suptitle("Feature Quality Analysis Summary", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"feature_quality_summary.{fmt}")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved to {output_path}")


def plot_evaluation_summary(results, output_dir, fmt="png"):
    """Create overall evaluation summary visualization."""
    print("Generating evaluation summary plot...")

    # Extract key metrics
    summary = results.get("summary", {})

    # Find best level results
    best_level_key = f"level_{summary.get('best_hierarchy_level', 0)}"
    level_data = results.get(best_level_key, {})

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Main metrics card
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis("off")

    main_metrics_text = f"""
H-JEPA Evaluation Summary
{'=' * 80}
Model: {results['metadata'].get('model_config', {}).get('encoder_type', 'N/A')}
Dataset: {results['metadata'].get('dataset', 'N/A').upper()}
Training: {results['metadata'].get('training_info', {}).get('epochs_trained', 'N/A')} epochs

Overall Status: {summary.get('overall_status', 'Unknown')}
Performance Tier: {summary.get('performance_tier', 'Unknown')}
Collapse Status: {summary.get('collapse_status', 'No collapse detected')}

Key Results:
  • Linear Probe Accuracy: {summary.get('best_linear_probe_accuracy', 0):.2f}%
  • k-NN Accuracy: {summary.get('best_knn_accuracy', 0):.2f}%
  • Effective Rank: {summary.get('effective_rank_mean', 0):.1f}
  • Best Level: {summary.get('best_hierarchy_level', 0)}
"""

    ax1.text(
        0.05,
        0.5,
        main_metrics_text,
        fontfamily="monospace",
        fontsize=11,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # 2. Linear probe and k-NN comparison
    ax2 = fig.add_subplot(gs[1, 0])
    methods = ["Linear\nProbe", "k-NN"]
    accuracies = [summary.get("best_linear_probe_accuracy", 0), summary.get("best_knn_accuracy", 0)]
    bars = ax2.bar(methods, accuracies, color=["steelblue", "coral"], alpha=0.8)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Primary Metrics", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Comparison to baselines (if available)
    ax3 = fig.add_subplot(gs[1, 1])
    if "comparison_to_baselines" in results:
        comp = results["comparison_to_baselines"]
        baseline_names = ["Random", "H-JEPA\n(Ours)", "Supervised"]
        baseline_accs = [
            comp.get("vs_random_init", {}).get("random_accuracy", 30),
            summary.get("best_linear_probe_accuracy", 0),
            comp.get("vs_supervised_upper_bound", {}).get("supervised_accuracy", 95),
        ]
        colors = ["red", "steelblue", "green"]
        bars = ax3.bar(baseline_names, baseline_accs, color=colors, alpha=0.7)
        for bar, acc in zip(bars, baseline_accs):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax3.set_ylabel("Accuracy (%)", fontsize=11)
        ax3.set_title("vs Baselines", fontsize=12, fontweight="bold")
        ax3.set_ylim(0, 100)
        ax3.grid(axis="y", alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No baseline\ncomparison",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )
        ax3.axis("off")

    # 4. Recommendations
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    recs = summary.get("recommendations", [])[:4]  # Top 4 recommendations
    recs_text = "Top Recommendations:\n\n" + "\n\n".join([f"• {rec}" for rec in recs])
    ax4.text(
        0.05,
        0.95,
        recs_text,
        fontsize=9,
        verticalalignment="top",
        wrap=True,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    # 5-7. Status indicators
    axes_bottom = [fig.add_subplot(gs[2, i]) for i in range(3)]

    # Status 1: Linear Probe Status
    ax = axes_bottom[0]
    linear_acc = summary.get("best_linear_probe_accuracy", 0)
    status = "✅ Excellent" if linear_acc >= 75 else "✅ Good" if linear_acc >= 70 else "⚠️ Moderate"
    color = "green" if linear_acc >= 75 else "blue" if linear_acc >= 70 else "orange"
    ax.text(
        0.5,
        0.5,
        f"{status}\n\nLinear Probe\n{linear_acc:.1f}%",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
    )
    ax.axis("off")

    # Status 2: Collapse Check
    ax = axes_bottom[1]
    collapse = summary.get("collapse_status", "Unknown")
    has_collapse = "collapse detected" in collapse.lower()
    status_text = "✅ Healthy" if not has_collapse else "❌ Collapse"
    color = "green" if not has_collapse else "red"
    ax.text(
        0.5,
        0.5,
        f"{status_text}\n\nRepresentations\n{collapse}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
    )
    ax.axis("off")

    # Status 3: Overall Status
    ax = axes_bottom[2]
    overall = summary.get("overall_status", "Unknown")
    tier = summary.get("performance_tier", "Unknown")
    ax.text(
        0.5,
        0.5,
        f"Overall Status\n\n{overall}\n\n{tier}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )
    ax.axis("off")

    plt.suptitle("H-JEPA Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98)

    output_path = os.path.join(output_dir, f"evaluation_dashboard.{fmt}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")


def main():
    args = parse_args()

    # Load results
    print(f"\nLoading evaluation results from {args.results}...")
    results = load_results(args.results)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    # Generate visualizations
    print("Generating visualizations...\n")

    plot_evaluation_summary(results, args.output_dir, args.format)
    plot_hierarchy_comparison(results, args.output_dir, args.format)
    plot_knn_sweep(results, args.output_dir, args.format)
    plot_per_class_accuracy(results, args.output_dir, args.format)
    plot_feature_quality_summary(results, args.output_dir, args.format)

    print(f"\n{'='*80}")
    print("Visualization generation complete!")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
