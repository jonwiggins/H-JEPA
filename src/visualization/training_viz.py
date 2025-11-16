"""
Training visualization utilities for H-JEPA.

Provides functions to visualize training curves, loss landscapes,
gradient flow, and collapse monitoring.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

try:
    import seaborn as sns
except ImportError:
    sns = None


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    smooth_window: int = 10,
) -> mfigure.Figure:
    """
    Plot training and validation curves.

    Args:
        metrics: Dictionary of metric lists (e.g., {'train_loss': [...], 'val_loss': [...]})
        save_path: Path to save figure
        figsize: Figure size
        smooth_window: Window size for smoothing curves

    Returns:
        Matplotlib figure
    """
    # Separate metrics by type
    loss_metrics = {k: v for k, v in metrics.items() if "loss" in k.lower()}
    other_metrics = {k: v for k, v in metrics.items() if "loss" not in k.lower()}

    num_plots = 2 if other_metrics else 1
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    # Plot 1: Loss curves
    for name, values in loss_metrics.items():
        if len(values) == 0:
            continue

        epochs = list(range(len(values)))
        axes[0].plot(epochs, values, label=name, alpha=0.3, linewidth=1)

        # Plot smoothed curve
        if len(values) >= smooth_window:
            smoothed = np.convolve(values, np.ones(smooth_window) / smooth_window, mode="valid")
            smooth_epochs = list(range(len(smoothed)))
            axes[0].plot(smooth_epochs, smoothed, label=f"{name} (smoothed)", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale("log")  # Log scale often better for loss

    # Plot 2: Other metrics
    if other_metrics:
        for name, values in other_metrics.items():
            if len(values) == 0:
                continue

            epochs = list(range(len(values)))
            axes[1].plot(
                epochs, values, label=name, alpha=0.7, linewidth=2, marker="o", markersize=3
            )

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Training Metrics")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_hierarchical_losses(
    hierarchical_losses: Dict[int, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> mfigure.Figure:
    """
    Plot losses for different hierarchical levels.

    Args:
        hierarchical_losses: Dictionary mapping level to loss values
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: All levels over time
    for level, losses in hierarchical_losses.items():
        epochs = list(range(len(losses)))
        axes[0].plot(epochs, losses, label=f"Level {level}", linewidth=2, marker="o", markersize=3)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Hierarchical Losses Over Time")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale("log")

    # Plot 2: Final loss per level
    final_losses = [losses[-1] if len(losses) > 0 else 0 for losses in hierarchical_losses.values()]
    levels = list(hierarchical_losses.keys())

    axes[1].bar(levels, final_losses, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Hierarchy Level")
    axes[1].set_ylabel("Final Loss")
    axes[1].set_title("Final Loss by Hierarchy Level")
    axes[1].grid(alpha=0.3, axis="y")

    for i, (level, loss) in enumerate(zip(levels, final_losses)):
        axes[1].text(level, loss, f"{loss:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_loss_landscape(
    model: nn.Module,
    dataloader: "torch.utils.data.DataLoader[Any]",
    criterion: nn.Module,
    directions: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    steps: int = 20,
    alpha_range: Tuple[float, float] = (-1.0, 1.0),
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> mfigure.Figure:
    """
    Visualize loss landscape around current parameters.

    Args:
        model: H-JEPA model
        dataloader: Data loader for computing loss
        criterion: Loss function
        directions: Two random directions for exploration (will be generated if None)
        steps: Number of steps in each direction
        alpha_range: Range of step sizes
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    model.eval()
    device = next(model.parameters()).device

    # Generate random directions if not provided
    if directions is None:
        direction1: List[torch.Tensor] = []
        direction2: List[torch.Tensor] = []

        for param in model.parameters():
            if param.requires_grad:
                d1 = torch.randn_like(param)
                d2 = torch.randn_like(param)

                # Normalize
                d1 = d1 / (torch.norm(d1) + 1e-8)
                d2 = d2 / (torch.norm(d2) + 1e-8)

                direction1.append(d1)
                direction2.append(d2)

        directions = (direction1, direction2)

    # Save original parameters
    original_params = [p.clone() for p in model.parameters() if p.requires_grad]

    # Create grid
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], steps)
    beta_vals = np.linspace(alpha_range[0], alpha_range[1], steps)

    losses = np.zeros((steps, steps))

    # Compute loss at each point
    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            # Update parameters
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad:
                    param.data = (
                        original_params[param_idx]
                        + alpha * directions[0][param_idx]
                        + beta * directions[1][param_idx]
                    )
                    param_idx += 1

            # Compute loss
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= 10:  # Limit to 10 batches for speed
                        break

                    images = (
                        batch[0].to(device)
                        if isinstance(batch, (list, tuple))
                        else batch.to(device)
                    )

                    # Generate mask (simplified)
                    B = images.shape[0]
                    num_patches: int = model.get_num_patches()  # type: ignore[operator]
                    mask_ratio = 0.5
                    num_masked = int(num_patches * mask_ratio)

                    mask = torch.zeros(B, num_patches, device=device)
                    for b in range(B):
                        masked_indices = torch.randperm(num_patches)[:num_masked]
                        mask[b, masked_indices] = 1

                    # Forward pass
                    output = model(images, mask, return_all_levels=False)

                    # Compute loss
                    loss = criterion(output["predictions"][0], output["targets"][0])
                    total_loss += loss.item()
                    num_batches += 1

            losses[i, j] = total_loss / max(num_batches, 1)

    # Restore original parameters
    param_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            param.data = original_params[param_idx]
            param_idx += 1

    # Visualize
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, subplot_kw={"projection": "3d"} if True else None
    )

    # Plot 1: Contour plot
    X, Y = np.meshgrid(alpha_vals, beta_vals)

    axes[0] = plt.subplot(1, 2, 1)
    contour = axes[0].contourf(X, Y, losses.T, levels=20, cmap="viridis")
    axes[0].contour(X, Y, losses.T, levels=10, colors="black", alpha=0.3, linewidths=0.5)
    axes[0].plot(0, 0, "r*", markersize=15, label="Current Position")
    axes[0].set_xlabel("Direction 1")
    axes[0].set_ylabel("Direction 2")
    axes[0].set_title("Loss Landscape (Contour)")
    axes[0].legend()
    plt.colorbar(contour, ax=axes[0], label="Loss")

    # Plot 2: 3D surface
    axes[1] = plt.subplot(1, 2, 2, projection="3d")
    surf = axes[1].plot_surface(X, Y, losses.T, cmap="viridis", alpha=0.8)
    axes[1].scatter([0], [0], [losses[steps // 2, steps // 2]], color="red", s=100, label="Current")
    axes[1].set_xlabel("Direction 1")
    axes[1].set_ylabel("Direction 2")
    axes[1].set_zlabel("Loss")
    axes[1].set_title("Loss Landscape (3D)")
    plt.colorbar(surf, ax=axes[1], label="Loss", shrink=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_gradient_flow(
    model: nn.Module,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> Optional[mfigure.Figure]:
    """
    Visualize gradient flow through the network.

    Args:
        model: H-JEPA model (after backward pass)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Collect gradient statistics
    layer_names = []
    mean_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layer_names.append(name)
            mean_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
            layers.append(param.grad.numel())

    if len(layer_names) == 0:
        print("Warning: No gradients found. Run backward pass before calling this function.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Mean gradient per layer
    axes[0, 0].barh(range(len(layer_names)), mean_grads, alpha=0.7, edgecolor="black")
    axes[0, 0].set_yticks(range(len(layer_names)))
    axes[0, 0].set_yticklabels([name.split(".")[-1] for name in layer_names], fontsize=7)
    axes[0, 0].set_xlabel("Mean Absolute Gradient")
    axes[0, 0].set_title("Mean Gradient per Layer")
    axes[0, 0].grid(alpha=0.3, axis="x")
    axes[0, 0].set_xscale("log")

    # Plot 2: Max gradient per layer
    axes[0, 1].barh(
        range(len(layer_names)), max_grads, alpha=0.7, edgecolor="black", color="orange"
    )
    axes[0, 1].set_yticks(range(len(layer_names)))
    axes[0, 1].set_yticklabels([name.split(".")[-1] for name in layer_names], fontsize=7)
    axes[0, 1].set_xlabel("Max Absolute Gradient")
    axes[0, 1].set_title("Max Gradient per Layer")
    axes[0, 1].grid(alpha=0.3, axis="x")
    axes[0, 1].set_xscale("log")

    # Plot 3: Gradient distribution (first few layers)
    sample_layers = min(5, len(layer_names))
    for i in range(sample_layers):
        name = layer_names[i]
        param = dict(model.named_parameters())[name]
        if param.grad is not None:
            grads = param.grad.cpu().flatten().numpy()
            # Sample if too large
            if len(grads) > 10000:
                grads = np.random.choice(grads, 10000, replace=False)
            axes[1, 0].hist(grads, bins=50, alpha=0.5, label=name.split(".")[-1])

    axes[1, 0].set_xlabel("Gradient Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Gradient Distribution (Sample Layers)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Gradient norm ratio (consecutive layers)
    if len(mean_grads) > 1:
        ratios = [mean_grads[i + 1] / (mean_grads[i] + 1e-10) for i in range(len(mean_grads) - 1)]
        axes[1, 1].plot(ratios, marker="o", markersize=4)
        axes[1, 1].axhline(y=1.0, color="r", linestyle="--", label="No change")
        axes[1, 1].set_xlabel("Layer Index")
        axes[1, 1].set_ylabel("Gradient Ratio")
        axes[1, 1].set_title("Gradient Flow Ratio (layer[i+1] / layer[i])")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_collapse_metrics(
    features: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> mfigure.Figure:
    """
    Plot metrics for detecting representational collapse.

    Args:
        features: Feature tensor [N, D]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    features_np: npt.NDArray[np.float64] = features.cpu().numpy()

    # Normalize features
    features_norm = features_np / (np.linalg.norm(features_np, axis=-1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Metric 1: Standard deviation collapse
    std_per_dim = features_norm.std(axis=0)

    axes[0].hist(std_per_dim, bins=50, alpha=0.7, edgecolor="black")
    axes[0].axvline(
        std_per_dim.mean(), color="red", linestyle="--", label=f"Mean: {std_per_dim.mean():.4f}"
    )
    axes[0].axvline(0.01, color="orange", linestyle="--", label="Collapse Threshold")
    axes[0].set_xlabel("Standard Deviation")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Dimension-wise Std Dev\n(Low std = potential collapse)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Metric 2: Rank of covariance matrix
    cov_matrix = np.cov(features_norm.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    # Compute effective rank
    eigenvalues_norm = eigenvalues / eigenvalues.sum()
    effective_rank = np.exp(-np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10)))

    axes[1].plot(eigenvalues, marker="o", markersize=3)
    axes[1].set_xlabel("Eigenvalue Index")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].set_title(
        f"Eigenvalue Spectrum\nEffective Rank: {effective_rank:.1f}/{len(eigenvalues)}"
    )
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale("log")

    # Metric 3: Pairwise cosine similarity
    # Sample for efficiency
    sample_size = min(1000, features_norm.shape[0])
    sample_indices = np.random.choice(features_norm.shape[0], sample_size, replace=False)
    features_sample = features_norm[sample_indices]

    # Compute pairwise similarities
    similarity_matrix = features_sample @ features_sample.T
    # Get upper triangle (exclude diagonal)
    triu_indices = np.triu_indices(sample_size, k=1)
    similarities = similarity_matrix[triu_indices]

    axes[2].hist(similarities, bins=50, alpha=0.7, edgecolor="black")
    axes[2].axvline(
        similarities.mean(), color="red", linestyle="--", label=f"Mean: {similarities.mean():.4f}"
    )
    axes[2].axvline(0.9, color="orange", linestyle="--", label="High Similarity Threshold")
    axes[2].set_xlabel("Cosine Similarity")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Pairwise Similarity Distribution\n(High sim = potential collapse)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # Add overall collapse warning
    collapse_indicators = []
    if std_per_dim.mean() < 0.01:
        collapse_indicators.append("Low std dev")
    if effective_rank < len(eigenvalues) * 0.1:
        collapse_indicators.append("Low rank")
    if similarities.mean() > 0.9:
        collapse_indicators.append("High similarity")

    if collapse_indicators:
        warning_text = "⚠ Collapse Warning: " + ", ".join(collapse_indicators)
        fig.text(
            0.5,
            0.02,
            warning_text,
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.3),
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ema_momentum(
    momentum_history: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> mfigure.Figure:
    """
    Plot EMA momentum schedule over training.

    Args:
        momentum_history: List of momentum values over training
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = list(range(len(momentum_history)))
    ax.plot(steps, momentum_history, linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("EMA Momentum")
    ax.set_title("Target Encoder EMA Momentum Schedule")
    ax.grid(alpha=0.3)

    # Add annotations
    if len(momentum_history) > 0:
        ax.axhline(
            y=momentum_history[0],
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Initial: {momentum_history[0]:.4f}",
        )
        ax.axhline(
            y=momentum_history[-1],
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Final: {momentum_history[-1]:.4f}",
        )
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def load_training_logs(log_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load training logs from directory.

    Args:
        log_dir: Directory containing training logs

    Returns:
        Dictionary of loaded metrics
    """
    log_dir = Path(log_dir)

    metrics = {}

    # Try to load JSON logs
    for json_file in log_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                metrics[json_file.stem] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    # Try to load numpy files
    for npy_file in log_dir.glob("*.npy"):
        try:
            data = np.load(npy_file, allow_pickle=True)
            metrics[npy_file.stem] = data.tolist() if isinstance(data, np.ndarray) else data
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")

    return metrics
