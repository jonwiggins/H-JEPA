"""
Feature quality metrics for H-JEPA representations.

This module implements various metrics to assess the quality of learned
representations, including rank analysis, variance measures, and isotropy.
"""

from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class FeatureQualityAnalyzer:
    """
    Analyzer for representation quality metrics.

    Args:
        model: H-JEPA model
        hierarchy_level: Which hierarchy level to analyze
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        hierarchy_level: int = 0,
        device: str = 'cuda',
    ):
        self.model = model
        self.hierarchy_level = hierarchy_level
        self.device = device

        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        pool: bool = True,
        normalize: bool = False,
        desc: str = "Extracting features"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataset.

        Args:
            dataloader: DataLoader for the dataset
            max_samples: Maximum number of samples to extract (None for all)
            pool: Whether to pool patch features
            normalize: Whether to normalize features
            desc: Progress bar description

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        all_features = []
        all_labels = []
        num_samples = 0

        for images, labels in tqdm(dataloader, desc=desc):
            if max_samples is not None and num_samples >= max_samples:
                break

            images = images.to(self.device)

            # Extract features at specified hierarchy level
            features = self.model.extract_features(
                images,
                level=self.hierarchy_level,
                use_target_encoder=True,
            )

            # Pool if requested
            if pool and features.ndim == 3:
                features = features.mean(dim=1)

            # Normalize if requested
            if normalize:
                features = F.normalize(features, p=2, dim=-1)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            num_samples += len(images)

            if max_samples is not None and num_samples >= max_samples:
                # Truncate last batch if needed
                excess = num_samples - max_samples
                if excess > 0:
                    all_features[-1] = all_features[-1][:-excess]
                    all_labels[-1] = all_labels[-1][:-excess]
                break

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, labels

    def compute_effective_rank(self, features: np.ndarray) -> float:
        """
        Compute effective rank of feature matrix using SVD.

        Effective rank measures how many dimensions are actually used by
        the representations. Higher is generally better (less collapse).

        Based on: Roy & Vetterli (2007) "The effective rank: A measure of
        effective dimensionality"

        Args:
            features: Feature matrix [N, D]

        Returns:
            Effective rank (between 1 and min(N, D))
        """
        # Center features
        features_centered = features - features.mean(axis=0, keepdims=True)

        # Compute SVD
        _, singular_values, _ = svd(features_centered, full_matrices=False)

        # Normalize singular values
        singular_values = singular_values / singular_values.sum()

        # Compute entropy
        entropy = -(singular_values * np.log(singular_values + 1e-12)).sum()

        # Effective rank is exp(entropy)
        effective_rank = np.exp(entropy)

        return effective_rank

    def compute_rank_analysis(
        self,
        features: np.ndarray,
        variance_threshold: float = 0.99
    ) -> Dict[str, float]:
        """
        Compute comprehensive rank analysis.

        Args:
            features: Feature matrix [N, D]
            variance_threshold: Variance threshold for dimensionality

        Returns:
            Dictionary with rank metrics
        """
        N, D = features.shape

        # Center features
        features_centered = features - features.mean(axis=0, keepdims=True)

        # Compute SVD
        _, singular_values, _ = svd(features_centered, full_matrices=False)

        # Normalize to get explained variance
        explained_variance = singular_values ** 2
        explained_variance = explained_variance / explained_variance.sum()

        # Cumulative variance
        cumulative_variance = np.cumsum(explained_variance)

        # Find number of components for threshold
        n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1

        # Effective rank
        effective_rank = self.compute_effective_rank(features)

        # Rank ratio (effective rank / feature dimension)
        rank_ratio = effective_rank / D

        metrics = {
            'effective_rank': effective_rank,
            'rank_ratio': rank_ratio,
            'feature_dim': D,
            'n_components_99': n_components,
            'variance_first_component': explained_variance[0],
            'variance_first_10': explained_variance[:10].sum(),
            'singular_value_max': singular_values[0],
            'singular_value_mean': singular_values.mean(),
        }

        return metrics

    def compute_feature_statistics(self, features: np.ndarray) -> Dict[str, float]:
        """
        Compute feature statistics (variance, covariance, etc.).

        Args:
            features: Feature matrix [N, D]

        Returns:
            Dictionary with statistics
        """
        # Center features
        features_centered = features - features.mean(axis=0, keepdims=True)

        # Per-dimension variance
        variances = features_centered.var(axis=0)

        # Covariance matrix
        cov_matrix = np.cov(features_centered.T)

        # Correlation matrix
        corr_matrix = np.corrcoef(features_centered.T)

        metrics = {
            'mean_variance': variances.mean(),
            'std_variance': variances.std(),
            'min_variance': variances.min(),
            'max_variance': variances.max(),
            'mean_feature': features.mean(),
            'std_feature': features.std(),
            'mean_covariance_off_diag': (cov_matrix - np.diag(np.diag(cov_matrix))).mean(),
            'mean_abs_correlation_off_diag': np.abs(corr_matrix - np.eye(corr_matrix.shape[0])).mean(),
        }

        return metrics

    def compute_isotropy(self, features: np.ndarray) -> Dict[str, float]:
        """
        Compute isotropy metrics.

        Isotropy measures whether features are uniformly distributed in all
        directions. Higher isotropy is generally better for downstream tasks.

        Args:
            features: Feature matrix [N, D]

        Returns:
            Dictionary with isotropy metrics
        """
        # Normalize features
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity matrix
        similarity_matrix = features_norm @ features_norm.T

        # Remove diagonal (self-similarity)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]

        # Isotropy metrics
        metrics = {
            'mean_similarity': similarities.mean(),
            'std_similarity': similarities.std(),
            'max_similarity': similarities.max(),
            'min_similarity': similarities.min(),
        }

        # Self-similarity (should be 1 for normalized features)
        self_similarities = np.diag(similarity_matrix)
        metrics['mean_self_similarity'] = self_similarities.mean()

        # Alignment and uniformity (from Wang & Isola, 2020)
        # Uniformity: how uniformly distributed features are
        # Lower is better (features spread out on hypersphere)
        uniformity = np.log(np.exp(2 * similarities).mean())
        metrics['uniformity'] = uniformity

        return metrics

    def detect_collapse(
        self,
        features: np.ndarray,
        threshold_rank_ratio: float = 0.1,
        threshold_variance: float = 0.01,
    ) -> Dict[str, bool]:
        """
        Detect representation collapse.

        Collapse occurs when the model outputs very similar features for
        different inputs, which is a failure mode in self-supervised learning.

        Args:
            features: Feature matrix [N, D]
            threshold_rank_ratio: Minimum acceptable rank ratio
            threshold_variance: Minimum acceptable mean variance

        Returns:
            Dictionary with collapse indicators
        """
        # Compute rank metrics
        rank_metrics = self.compute_rank_analysis(features)

        # Compute variance
        feature_stats = self.compute_feature_statistics(features)

        # Check for collapse
        collapse_indicators = {
            'rank_collapse': rank_metrics['rank_ratio'] < threshold_rank_ratio,
            'variance_collapse': feature_stats['mean_variance'] < threshold_variance,
            'dimension_collapse': rank_metrics['n_components_99'] < features.shape[1] * 0.1,
        }

        # Overall collapse
        collapse_indicators['any_collapse'] = any(collapse_indicators.values())

        return collapse_indicators

    def compute_all_metrics(
        self,
        dataloader: DataLoader,
        max_samples: int = 10000,
    ) -> Dict[str, Dict]:
        """
        Compute all feature quality metrics.

        Args:
            dataloader: DataLoader for the dataset
            max_samples: Maximum samples to use for analysis

        Returns:
            Dictionary with all metrics
        """
        # Extract features
        features, labels = self.extract_features(
            dataloader,
            max_samples=max_samples,
            pool=True,
            normalize=False,
            desc="Extracting features for analysis"
        )

        print(f"Analyzing {len(features)} samples with {features.shape[1]} dimensions")

        # Compute all metrics
        rank_metrics = self.compute_rank_analysis(features)
        feature_stats = self.compute_feature_statistics(features)
        isotropy_metrics = self.compute_isotropy(features)
        collapse_indicators = self.detect_collapse(features)

        results = {
            'rank': rank_metrics,
            'statistics': feature_stats,
            'isotropy': isotropy_metrics,
            'collapse': collapse_indicators,
            'num_samples': len(features),
            'feature_dim': features.shape[1],
        }

        return results

    def visualize_features_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Visualize features using t-SNE.

        Args:
            features: Feature matrix [N, D]
            labels: Labels [N]
            n_components: Number of t-SNE dimensions
            perplexity: t-SNE perplexity
            random_state: Random seed

        Returns:
            t-SNE embeddings [N, n_components]
        """
        print(f"Computing t-SNE with perplexity={perplexity}...")

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            verbose=1,
        )

        embeddings = tsne.fit_transform(features)

        return embeddings

    def visualize_features_umap(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Visualize features using UMAP.

        Args:
            features: Feature matrix [N, D]
            labels: Labels [N]
            n_components: Number of UMAP dimensions
            n_neighbors: Number of neighbors
            min_dist: Minimum distance
            random_state: Random seed

        Returns:
            UMAP embeddings [N, n_components]
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")

        print(f"Computing UMAP with n_neighbors={n_neighbors}...")

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            verbose=True,
        )

        embeddings = reducer.fit_transform(features)

        return embeddings

    def compute_pca(
        self,
        features: np.ndarray,
        n_components: int = 50,
    ) -> Tuple[np.ndarray, PCA]:
        """
        Compute PCA of features.

        Args:
            features: Feature matrix [N, D]
            n_components: Number of PCA components

        Returns:
            Tuple of (PCA embeddings, PCA object)
        """
        print(f"Computing PCA with {n_components} components...")

        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(features)

        print(f"Explained variance ratio: {pca.explained_variance_ratio_[:10]}")
        print(f"Cumulative variance: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

        return embeddings, pca


def analyze_feature_quality(
    model: nn.Module,
    dataloader: DataLoader,
    hierarchy_level: int = 0,
    max_samples: int = 10000,
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Convenience function for feature quality analysis.

    Args:
        model: H-JEPA model
        dataloader: Data loader
        hierarchy_level: Which hierarchy level to analyze
        max_samples: Maximum samples for analysis
        device: Device to run on

    Returns:
        Dictionary with all metrics
    """
    analyzer = FeatureQualityAnalyzer(
        model=model,
        hierarchy_level=hierarchy_level,
        device=device,
    )

    results = analyzer.compute_all_metrics(
        dataloader=dataloader,
        max_samples=max_samples,
    )

    return results


def print_quality_report(metrics: Dict[str, Dict], verbose: bool = True):
    """
    Print a formatted report of feature quality metrics.

    Args:
        metrics: Metrics dictionary from compute_all_metrics
        verbose: Whether to print detailed metrics
    """
    print("\n" + "="*60)
    print("FEATURE QUALITY REPORT")
    print("="*60)

    # Basic info
    print(f"\nDataset: {metrics['num_samples']} samples, {metrics['feature_dim']} dimensions")

    # Rank analysis
    print("\n--- Rank Analysis ---")
    rank = metrics['rank']
    print(f"Effective Rank: {rank['effective_rank']:.2f}")
    print(f"Rank Ratio: {rank['rank_ratio']:.4f}")
    print(f"Components for 99% variance: {rank['n_components_99']}")
    print(f"First component variance: {rank['variance_first_component']:.4f}")

    # Feature statistics
    print("\n--- Feature Statistics ---")
    stats = metrics['statistics']
    print(f"Mean variance: {stats['mean_variance']:.6f}")
    print(f"Std variance: {stats['std_variance']:.6f}")
    print(f"Mean |correlation|: {stats['mean_abs_correlation_off_diag']:.6f}")

    # Isotropy
    print("\n--- Isotropy ---")
    iso = metrics['isotropy']
    print(f"Mean similarity: {iso['mean_similarity']:.6f}")
    print(f"Std similarity: {iso['std_similarity']:.6f}")
    print(f"Uniformity (lower is better): {iso['uniformity']:.6f}")

    # Collapse detection
    print("\n--- Collapse Detection ---")
    collapse = metrics['collapse']
    if collapse['any_collapse']:
        print("WARNING: Potential representation collapse detected!")
        if collapse['rank_collapse']:
            print("  - Rank collapse: effective rank is too low")
        if collapse['variance_collapse']:
            print("  - Variance collapse: feature variance is too low")
        if collapse['dimension_collapse']:
            print("  - Dimension collapse: too few dimensions used")
    else:
        print("No collapse detected - representations look healthy!")

    if verbose:
        print("\n--- Detailed Metrics ---")
        print("Rank metrics:", rank)
        print("Statistics:", stats)
        print("Isotropy:", iso)

    print("\n" + "="*60)


def compare_hierarchy_levels(
    model: nn.Module,
    dataloader: DataLoader,
    num_levels: Optional[int] = None,
    max_samples: int = 10000,
    device: str = 'cuda',
) -> Dict[int, Dict]:
    """
    Compare feature quality across hierarchy levels.

    Args:
        model: H-JEPA model
        dataloader: Data loader
        num_levels: Number of hierarchy levels (None to use model's num_hierarchies)
        max_samples: Maximum samples for analysis
        device: Device to run on

    Returns:
        Dictionary mapping level to metrics
    """
    if num_levels is None:
        num_levels = model.num_hierarchies

    results = {}

    for level in range(num_levels):
        print(f"\n{'='*60}")
        print(f"Analyzing Hierarchy Level {level}")
        print(f"{'='*60}")

        analyzer = FeatureQualityAnalyzer(
            model=model,
            hierarchy_level=level,
            device=device,
        )

        metrics = analyzer.compute_all_metrics(
            dataloader=dataloader,
            max_samples=max_samples,
        )

        results[level] = metrics

        # Print summary
        print(f"\nLevel {level} Summary:")
        print(f"  Effective Rank: {metrics['rank']['effective_rank']:.2f}")
        print(f"  Rank Ratio: {metrics['rank']['rank_ratio']:.4f}")
        print(f"  Mean Variance: {metrics['statistics']['mean_variance']:.6f}")
        print(f"  Uniformity: {metrics['isotropy']['uniformity']:.6f}")
        print(f"  Collapse: {metrics['collapse']['any_collapse']}")

    return results
