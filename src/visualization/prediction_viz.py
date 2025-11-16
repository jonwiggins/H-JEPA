"""
Prediction visualization utilities for H-JEPA.

Provides functions to visualize predictions vs ground truth,
feature space embeddings, nearest neighbors, and reconstruction quality.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    TSNE = None


def visualize_predictions(
    model: nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    original_image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """
    Visualize model predictions for masked regions.

    Args:
        model: H-JEPA model
        image: Input image [1, C, H, W]
        mask: Binary mask [N] where 1 = masked
        original_image: Original image for display [H, W, 3]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    model.eval()

    with torch.no_grad():
        output = model(image, mask, return_all_levels=True)

    predictions = output["predictions"]
    targets = output["targets"]

    # Use finest hierarchy level
    pred = predictions[0][0]  # [num_masked, embed_dim]
    target = targets[0][0]  # [num_masked, embed_dim]

    # Compute prediction error
    error = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # [num_masked]

    # Create mask visualization
    mask_np = mask[0].cpu().numpy()
    grid_size = int(np.sqrt(len(mask_np)))

    # Create error map
    error_map = np.zeros(len(mask_np))
    masked_indices = np.where(mask_np > 0.5)[0]
    error_map[masked_indices] = error.cpu().numpy()
    error_map_2d = error_map.reshape(grid_size, grid_size)

    # Create figure
    num_plots = 5 if original_image is not None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    plot_idx = 0

    # Plot 1: Original image (if available)
    if original_image is not None:
        axes[plot_idx].imshow(original_image)
        axes[plot_idx].set_title("Original Image")
        axes[plot_idx].axis("off")
        plot_idx += 1

    # Plot 2: Mask
    mask_2d = mask_np.reshape(grid_size, grid_size)
    axes[plot_idx].imshow(mask_2d, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
    axes[plot_idx].set_title(f"Mask\n({mask_np.mean():.1%} masked)")
    axes[plot_idx].axis("off")
    plot_idx += 1

    # Plot 3: Prediction error heatmap
    im2 = axes[plot_idx].imshow(error_map_2d, cmap="hot", interpolation="bilinear")
    axes[plot_idx].set_title("Prediction Error\n(MSE per patch)")
    axes[plot_idx].axis("off")
    plt.colorbar(im2, ax=axes[plot_idx], fraction=0.046, pad=0.04)
    plot_idx += 1

    # Plot 4: Cosine similarity
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # [num_masked]
    cos_sim_map = np.zeros(len(mask_np))
    cos_sim_map[masked_indices] = cos_sim.cpu().numpy()
    cos_sim_map_2d = cos_sim_map.reshape(grid_size, grid_size)

    im3 = axes[plot_idx].imshow(
        cos_sim_map_2d, cmap="RdYlGn", vmin=-1, vmax=1, interpolation="bilinear"
    )
    axes[plot_idx].set_title("Cosine Similarity\n(Prediction vs Target)")
    axes[plot_idx].axis("off")
    plt.colorbar(im3, ax=axes[plot_idx], fraction=0.046, pad=0.04)
    plot_idx += 1

    # Plot 5: Statistics
    stats_text = "Prediction Statistics:\n\n"
    stats_text += f"Masked patches: {len(masked_indices)}\n"
    stats_text += f"Mean error: {error.mean().item():.4f}\n"
    stats_text += f"Std error: {error.std().item():.4f}\n"
    stats_text += f"Mean cos sim: {cos_sim.mean().item():.4f}\n"
    stats_text += f"Min cos sim: {cos_sim.min().item():.4f}\n"
    stats_text += f"Max cos sim: {cos_sim.max().item():.4f}"

    axes[plot_idx].text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )
    axes[plot_idx].axis("off")
    axes[plot_idx].set_title("Statistics")

    plt.suptitle("H-JEPA Predictions vs Targets", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_hierarchical_predictions(
    model: nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """
    Visualize predictions at different hierarchical levels.

    Args:
        model: H-JEPA model
        image: Input image [1, C, H, W]
        mask: Binary mask [N] where 1 = masked
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    model.eval()

    with torch.no_grad():
        output = model(image, mask, return_all_levels=True)

    predictions = output["predictions"]
    targets = output["targets"]
    num_levels = len(predictions)

    fig, axes = plt.subplots(num_levels, 3, figsize=figsize, squeeze=False)

    for level in range(num_levels):
        pred = predictions[level][0]  # [num_patches_level, embed_dim]
        target = targets[level][0]  # [num_patches_level, embed_dim]

        # Compute metrics
        mse_error = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        cos_sim = F.cosine_similarity(pred, target, dim=-1)

        # Plot 1: MSE error
        axes[level, 0].plot(mse_error.cpu().numpy(), marker="o", markersize=3)
        axes[level, 0].set_title(f"Level {level}: MSE Error")
        axes[level, 0].set_xlabel("Patch Index")
        axes[level, 0].set_ylabel("MSE")
        axes[level, 0].grid(alpha=0.3)

        # Plot 2: Cosine similarity
        axes[level, 1].plot(cos_sim.cpu().numpy(), marker="o", markersize=3, color="green")
        axes[level, 1].set_title(f"Level {level}: Cosine Similarity")
        axes[level, 1].set_xlabel("Patch Index")
        axes[level, 1].set_ylabel("Cosine Similarity")
        axes[level, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[level, 1].grid(alpha=0.3)

        # Plot 3: Statistics
        stats_text = f"Level {level} Statistics:\n\n"
        stats_text += f"Patches: {len(pred)}\n"
        stats_text += f"Embed dim: {pred.shape[-1]}\n"
        stats_text += f"Mean MSE: {mse_error.mean().item():.4f}\n"
        stats_text += f"Mean cos sim: {cos_sim.mean().item():.4f}\n"

        axes[level, 2].text(
            0.1,
            0.5,
            stats_text,
            fontsize=9,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
        axes[level, 2].axis("off")

    plt.suptitle("Hierarchical Predictions Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_feature_space(
    features: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    **kwargs,
) -> plt.Figure:
    """
    Visualize high-dimensional features using dimensionality reduction.

    Args:
        features: Feature tensor [N, D]
        labels: Optional labels for coloring points [N]
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        save_path: Path to save figure
        figsize: Figure size
        **kwargs: Additional arguments for reduction method

    Returns:
        Matplotlib figure
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for feature space visualization. "
            "Install with: pip install scikit-learn"
        )

    # Convert to numpy
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features

    # Apply dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, **kwargs)
        features_2d = reducer.fit_transform(features_np)
        method_name = "t-SNE"
    elif method.lower() == "pca":
        reducer = PCA(n_components=2, **kwargs)
        features_2d = reducer.fit_transform(features_np)
        method_name = "PCA"
    elif method.lower() == "umap":
        try:
            from umap import UMAP

            reducer = UMAP(n_components=2, **kwargs)
            features_2d = reducer.fit_transform(features_np)
            method_name = "UMAP"
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            reducer = TSNE(n_components=2, **kwargs)
            features_2d = reducer.fit_transform(features_np)
            method_name = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Scatter plot
    if labels is not None:
        scatter = axes[0].scatter(
            features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab20", alpha=0.6, s=20
        )
        plt.colorbar(scatter, ax=axes[0])
    else:
        axes[0].scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=20, c="blue")

    axes[0].set_title(f"{method_name} Visualization")
    axes[0].set_xlabel(f"{method_name} Component 1")
    axes[0].set_ylabel(f"{method_name} Component 2")
    axes[0].grid(alpha=0.3)

    # Plot 2: Density plot
    from scipy.stats import gaussian_kde

    # Calculate point density
    xy = features_2d.T
    z = gaussian_kde(xy)(xy)

    scatter2 = axes[1].scatter(
        features_2d[:, 0], features_2d[:, 1], c=z, s=20, cmap="viridis", alpha=0.6
    )
    axes[1].set_title(f"{method_name} Density")
    axes[1].set_xlabel(f"{method_name} Component 1")
    axes[1].set_ylabel(f"{method_name} Component 2")
    axes[1].grid(alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label="Density")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_nearest_neighbors(
    model: nn.Module,
    query_image: torch.Tensor,
    database_images: torch.Tensor,
    database_labels: Optional[List[str]] = None,
    k: int = 5,
    level: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """
    Visualize nearest neighbors in embedding space.

    Args:
        model: H-JEPA model
        query_image: Query image [1, C, H, W]
        database_images: Database images [N, C, H, W]
        database_labels: Optional labels for database images
        k: Number of nearest neighbors to show
        level: Hierarchy level for features
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    model.eval()

    with torch.no_grad():
        # Extract features
        query_features = model.extract_features(query_image, level=level)
        database_features = model.extract_features(database_images, level=level)

        # Pool features (mean pooling)
        query_feat = query_features.mean(dim=1)  # [1, embed_dim]
        db_feat = database_features.mean(dim=1)  # [N, embed_dim]

        # Compute similarities
        similarities = F.cosine_similarity(query_feat, db_feat, dim=-1)  # [N]

        # Get top-k
        top_k_values, top_k_indices = torch.topk(similarities, k)

    # Visualize
    fig, axes = plt.subplots(1, k + 1, figsize=figsize)

    # Plot query image
    query_img = query_image[0].cpu().permute(1, 2, 0).numpy()
    if query_img.max() > 1.0:
        query_img = query_img / 255.0

    axes[0].imshow(query_img)
    axes[0].set_title("Query Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    axes[0].add_patch(
        Rectangle(
            (0, 0),
            query_img.shape[1] - 1,
            query_img.shape[0] - 1,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
    )

    # Plot nearest neighbors
    for i in range(k):
        idx = top_k_indices[i].item()
        sim = top_k_values[i].item()

        img = database_images[idx].cpu().permute(1, 2, 0).numpy()
        if img.max() > 1.0:
            img = img / 255.0

        axes[i + 1].imshow(img)

        title = f"#{i + 1}: sim={sim:.3f}"
        if database_labels is not None:
            title += f"\n{database_labels[idx]}"

        axes[i + 1].set_title(title, fontsize=9)
        axes[i + 1].axis("off")

    plt.suptitle(f"Nearest Neighbors (Level {level})", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_reconstruction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    original_image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """
    Visualize reconstruction quality (if using a decoder).

    Args:
        predictions: Predicted features/pixels [B, N, D] or [B, C, H, W]
        targets: Target features/pixels [B, N, D] or [B, C, H, W]
        mask: Binary mask [B, N] where 1 = masked
        original_image: Original image for reference [H, W, 3]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions[0].cpu().numpy()
    else:
        pred_np = predictions[0]

    if isinstance(targets, torch.Tensor):
        target_np = targets[0].cpu().numpy()
    else:
        target_np = targets[0]

    if isinstance(mask, torch.Tensor):
        mask_np = mask[0].cpu().numpy()
    else:
        mask_np = mask[0]

    # Compute reconstruction error
    if len(pred_np.shape) == 2:  # Feature space [N, D]
        error = np.linalg.norm(pred_np - target_np, axis=-1)  # [N]
        grid_size = int(np.sqrt(len(error)))
        error_2d = error.reshape(grid_size, grid_size)
    else:  # Pixel space [C, H, W]
        error = np.abs(pred_np - target_np).mean(axis=0)  # [H, W]
        error_2d = error

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Plot 1: Original image
    if original_image is not None:
        axes[0].imshow(original_image)
        axes[0].set_title("Original")
        axes[0].axis("off")
    else:
        axes[0].text(0.5, 0.5, "Original\nNot Available", ha="center", va="center", fontsize=12)
        axes[0].axis("off")

    # Plot 2: Target
    if len(target_np.shape) == 3 and target_np.shape[0] in [1, 3]:  # Image
        target_img = np.transpose(target_np, (1, 2, 0))
        if target_img.shape[-1] == 1:
            target_img = target_img.squeeze(-1)
        axes[1].imshow(target_img, cmap="gray" if len(target_img.shape) == 2 else None)
    else:  # Feature map
        axes[1].text(0.5, 0.5, "Target\nFeatures", ha="center", va="center", fontsize=12)
    axes[1].set_title("Target")
    axes[1].axis("off")

    # Plot 3: Prediction
    if len(pred_np.shape) == 3 and pred_np.shape[0] in [1, 3]:  # Image
        pred_img = np.transpose(pred_np, (1, 2, 0))
        if pred_img.shape[-1] == 1:
            pred_img = pred_img.squeeze(-1)
        axes[2].imshow(pred_img, cmap="gray" if len(pred_img.shape) == 2 else None)
    else:  # Feature map
        axes[2].text(0.5, 0.5, "Predicted\nFeatures", ha="center", va="center", fontsize=12)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # Plot 4: Error map
    im = axes[3].imshow(error_2d, cmap="hot", interpolation="bilinear")
    axes[3].set_title("Reconstruction Error")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle("Reconstruction Quality Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_embedding_distribution(
    features: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Visualize the distribution of embedding features.

    Args:
        features: Feature tensor [N, D]
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Feature magnitude distribution
    magnitudes = np.linalg.norm(features_np, axis=-1)
    axes[0].hist(magnitudes, bins=50, alpha=0.7, edgecolor="black")
    axes[0].axvline(
        magnitudes.mean(), color="red", linestyle="--", label=f"Mean: {magnitudes.mean():.2f}"
    )
    axes[0].set_xlabel("Feature Magnitude")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Embedding Magnitude Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Dimension-wise variance
    dim_variance = features_np.var(axis=0)
    axes[1].plot(dim_variance, alpha=0.7)
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Variance")
    axes[1].set_title("Variance per Dimension")
    axes[1].grid(alpha=0.3)

    # Plot 3: Correlation matrix (sample)
    sample_size = min(100, features_np.shape[-1])
    sample_indices = np.random.choice(features_np.shape[-1], sample_size, replace=False)
    corr_matrix = np.corrcoef(features_np[:, sample_indices].T)

    im = axes[2].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_title(f"Correlation Matrix\n(sample {sample_size} dims)")
    axes[2].set_xlabel("Dimension")
    axes[2].set_ylabel("Dimension")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
