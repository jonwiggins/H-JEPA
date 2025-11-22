"""
Unit tests for feature quality analysis module.

Tests for:
1. FeatureQualityAnalyzer initialization
2. Feature extraction
3. Effective rank computation
4. Rank analysis
5. Feature statistics
6. Isotropy metrics
7. Collapse detection
8. PCA computation
9. t-SNE/UMAP visualization prep
10. Edge cases
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.feature_quality import (
    FeatureQualityAnalyzer,
    analyze_feature_quality,
    compare_hierarchy_levels,
    print_quality_report,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model():
    """Create a mock H-JEPA model for testing."""
    mock = MagicMock()
    mock.embed_dim = 384
    mock.num_hierarchies = 3

    def mock_extract_features(images, level=0, use_target_encoder=True):
        batch_size = images.shape[0]
        # Return patch features [B, N, D]
        return torch.randn(batch_size, 196, 384)

    mock.extract_features = mock_extract_features
    mock.eval = MagicMock(return_value=mock)
    mock.parameters = MagicMock(return_value=[])

    return mock


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader."""
    num_samples = 100
    num_classes = 10
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.fixture
def small_dataloader():
    """Create a small dataloader for quick tests."""
    num_samples = 32
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 5, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


# ============================================================================
# Test Initialization
# ============================================================================


def test_analyzer_initialization(mock_model):
    """Test that FeatureQualityAnalyzer initializes correctly."""
    analyzer = FeatureQualityAnalyzer(
        model=mock_model,
        hierarchy_level=0,
        device="cpu",
    )

    assert analyzer.model is mock_model
    assert analyzer.hierarchy_level == 0
    assert analyzer.device == "cpu"


def test_analyzer_freezes_model(mock_model):
    """Test that model is frozen during initialization."""
    param = nn.Parameter(torch.randn(10, 10))
    param.requires_grad = True
    mock_model.parameters = MagicMock(return_value=[param])

    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    mock_model.eval.assert_called()
    assert not param.requires_grad


# ============================================================================
# Test Feature Extraction
# ============================================================================


def test_extract_features_basic(mock_model, small_dataloader, random_seed):
    """Test basic feature extraction."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features, labels = analyzer.extract_features(small_dataloader, pool=True, normalize=False)

    assert features.shape[0] == 32  # num_samples
    assert features.shape[1] == 384  # embed_dim
    assert labels.shape[0] == 32
    assert isinstance(features, np.ndarray)
    assert isinstance(labels, np.ndarray)


def test_extract_features_with_pooling(mock_model, small_dataloader, random_seed):
    """Test feature extraction with pooling."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features, labels = analyzer.extract_features(small_dataloader, pool=True, normalize=False)

    # Should be 2D after pooling
    assert features.ndim == 2
    assert features.shape[1] == 384


def test_extract_features_without_pooling(mock_model, random_seed):
    """Test feature extraction without pooling."""
    # Create model that returns 3D features
    mock_3d = MagicMock()
    mock_3d.embed_dim = 384

    def mock_extract_3d(images, level=0, use_target_encoder=True):
        batch_size = images.shape[0]
        return torch.randn(batch_size, 196, 384)  # [B, N, D]

    mock_3d.extract_features = mock_extract_3d
    mock_3d.eval = MagicMock(return_value=mock_3d)
    mock_3d.parameters = MagicMock(return_value=[])

    analyzer = FeatureQualityAnalyzer(model=mock_3d, device="cpu")

    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 5, (8,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8)

    features, _ = analyzer.extract_features(dataloader, pool=False)

    # Should be 3D without pooling
    assert features.ndim == 3
    assert features.shape == (8, 196, 384)


def test_extract_features_with_normalization(mock_model, small_dataloader, random_seed):
    """Test feature extraction with normalization."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features, _ = analyzer.extract_features(small_dataloader, pool=True, normalize=True)

    # Check features are approximately normalized
    norms = np.linalg.norm(features, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_extract_features_max_samples(mock_model, simple_dataloader, random_seed):
    """Test feature extraction with max_samples limit."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features, labels = analyzer.extract_features(simple_dataloader, max_samples=50)

    assert features.shape[0] == 50
    assert labels.shape[0] == 50


# ============================================================================
# Test Effective Rank
# ============================================================================


def test_compute_effective_rank_basic(mock_model, random_seed):
    """Test effective rank computation."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # Create random features
    features = np.random.randn(100, 50)

    effective_rank = analyzer.compute_effective_rank(features)

    assert isinstance(effective_rank, float)
    assert 1.0 <= effective_rank <= min(100, 50)


def test_compute_effective_rank_full_rank(mock_model, random_seed):
    """Test effective rank with full rank matrix."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # Create random full-rank matrix
    features = np.random.randn(100, 50)

    effective_rank = analyzer.compute_effective_rank(features)

    # Should be close to min dimension for random matrix
    assert effective_rank > 30  # Reasonably high rank


def test_compute_effective_rank_low_rank(mock_model, random_seed):
    """Test effective rank with low rank matrix."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # Create low-rank matrix (rank 5)
    U = np.random.randn(100, 5)
    V = np.random.randn(5, 50)
    features = U @ V

    effective_rank = analyzer.compute_effective_rank(features)

    # Should be close to 5
    assert effective_rank < 10  # Low rank


def test_compute_effective_rank_collapsed(mock_model, random_seed):
    """Test effective rank with collapsed features (all same)."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # All features are the same (collapsed)
    features = np.ones((100, 50))

    effective_rank = analyzer.compute_effective_rank(features)

    # With zero variance, SVD returns all zeros, resulting in NaN
    # This is expected for completely collapsed features
    assert np.isnan(effective_rank) or effective_rank < 2


# ============================================================================
# Test Rank Analysis
# ============================================================================


def test_compute_rank_analysis(mock_model, random_seed):
    """Test comprehensive rank analysis."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(100, 50)

    metrics = analyzer.compute_rank_analysis(features)

    assert "effective_rank" in metrics
    assert "rank_ratio" in metrics
    assert "feature_dim" in metrics
    assert "n_components_99" in metrics
    assert "variance_first_component" in metrics
    assert "variance_first_10" in metrics
    assert "singular_value_max" in metrics
    assert "singular_value_mean" in metrics

    assert metrics["feature_dim"] == 50
    assert 0 < metrics["rank_ratio"] <= 1


def test_compute_rank_analysis_variance_threshold(mock_model, random_seed):
    """Test rank analysis with different variance thresholds."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(100, 50)

    metrics_99 = analyzer.compute_rank_analysis(features, variance_threshold=0.99)
    metrics_95 = analyzer.compute_rank_analysis(features, variance_threshold=0.95)

    # Should need fewer components for 95% variance
    assert metrics_95["n_components_99"] <= metrics_99["n_components_99"]


# ============================================================================
# Test Feature Statistics
# ============================================================================


def test_compute_feature_statistics(mock_model, random_seed):
    """Test feature statistics computation."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(100, 50)

    stats = analyzer.compute_feature_statistics(features)

    assert "mean_variance" in stats
    assert "std_variance" in stats
    assert "min_variance" in stats
    assert "max_variance" in stats
    assert "mean_feature" in stats
    assert "std_feature" in stats
    assert "mean_covariance_off_diag" in stats
    assert "mean_abs_correlation_off_diag" in stats

    # Sanity checks
    assert stats["mean_variance"] > 0
    assert stats["min_variance"] <= stats["mean_variance"] <= stats["max_variance"]


def test_compute_feature_statistics_zero_variance(mock_model, random_seed):
    """Test statistics with zero variance features."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # All features are constant
    features = np.ones((100, 50))

    stats = analyzer.compute_feature_statistics(features)

    # Variance should be essentially zero
    assert stats["mean_variance"] < 1e-10
    assert stats["min_variance"] < 1e-10


# ============================================================================
# Test Isotropy
# ============================================================================


def test_compute_isotropy(mock_model, random_seed):
    """Test isotropy metrics computation."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(100, 50)

    isotropy = analyzer.compute_isotropy(features)

    assert "mean_similarity" in isotropy
    assert "std_similarity" in isotropy
    assert "max_similarity" in isotropy
    assert "min_similarity" in isotropy
    assert "mean_self_similarity" in isotropy
    assert "uniformity" in isotropy

    # Self-similarity should be close to 1 for normalized features
    assert abs(isotropy["mean_self_similarity"] - 1.0) < 0.1


def test_compute_isotropy_normalized_features(mock_model, random_seed):
    """Test isotropy with normalized features."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # Create normalized features
    features = np.random.randn(100, 50)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    isotropy = analyzer.compute_isotropy(features)

    # Self-similarity should be exactly 1
    assert np.allclose(isotropy["mean_self_similarity"], 1.0, atol=1e-5)


def test_compute_isotropy_identical_features(mock_model, random_seed):
    """Test isotropy with identical features (collapse)."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # All features are the same
    features = np.ones((100, 50))

    isotropy = analyzer.compute_isotropy(features)

    # All similarities should be 1 (perfect alignment)
    assert np.allclose(isotropy["mean_similarity"], 1.0, atol=1e-5)
    assert isotropy["std_similarity"] < 1e-5


# ============================================================================
# Test Collapse Detection
# ============================================================================


def test_detect_collapse_healthy(mock_model, random_seed):
    """Test collapse detection with healthy features."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # Create healthy random features
    features = np.random.randn(100, 50)

    collapse = analyzer.detect_collapse(features)

    assert "rank_collapse" in collapse
    assert "variance_collapse" in collapse
    assert "dimension_collapse" in collapse
    assert "any_collapse" in collapse

    # Should not detect collapse with random features
    assert not collapse["any_collapse"]


def test_detect_collapse_rank_collapse(mock_model, random_seed):
    """Test detection of rank collapse."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # Create very low-rank features
    U = np.random.randn(100, 2)
    V = np.random.randn(2, 50)
    features = U @ V

    collapse = analyzer.detect_collapse(features, threshold_rank_ratio=0.1)

    # Should detect rank collapse
    assert collapse["rank_collapse"]
    assert collapse["any_collapse"]


def test_detect_collapse_variance_collapse(mock_model, random_seed):
    """Test detection of variance collapse."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # Create features with very low variance
    features = np.random.randn(100, 50) * 0.001

    collapse = analyzer.detect_collapse(features, threshold_variance=0.01)

    # Should detect variance collapse
    assert collapse["variance_collapse"]
    assert collapse["any_collapse"]


def test_detect_collapse_custom_thresholds(mock_model, random_seed):
    """Test collapse detection with custom thresholds."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(100, 50)

    # Very strict thresholds
    collapse_strict = analyzer.detect_collapse(
        features,
        threshold_rank_ratio=0.9,
        threshold_variance=1.0,
    )

    # Lenient thresholds
    collapse_lenient = analyzer.detect_collapse(
        features,
        threshold_rank_ratio=0.01,
        threshold_variance=0.001,
    )

    # Strict should detect collapse more often
    assert collapse_strict["any_collapse"] or not collapse_lenient["any_collapse"]


# ============================================================================
# Test PCA
# ============================================================================


def test_compute_pca(mock_model, random_seed):
    """Test PCA computation."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(100, 50)

    embeddings, pca = analyzer.compute_pca(features, n_components=10)

    assert embeddings.shape == (100, 10)
    assert len(pca.explained_variance_ratio_) == 10
    assert 0 <= pca.explained_variance_ratio_.sum() <= 1.0


def test_compute_pca_full_components(mock_model, random_seed):
    """Test PCA with all components."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(100, 50)

    embeddings, pca = analyzer.compute_pca(features, n_components=50)

    assert embeddings.shape == (100, 50)
    assert len(pca.explained_variance_ratio_) == 50


# ============================================================================
# Test t-SNE Visualization
# ============================================================================


def test_visualize_features_tsne(mock_model, random_seed):
    """Test t-SNE visualization."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(50, 20)
    labels = np.random.randint(0, 5, 50)

    embeddings = analyzer.visualize_features_tsne(features, labels, n_components=2, random_state=42)

    assert embeddings.shape == (50, 2)
    assert isinstance(embeddings, np.ndarray)


def test_visualize_features_tsne_3d(mock_model, random_seed):
    """Test 3D t-SNE visualization."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(50, 20)
    labels = np.random.randint(0, 5, 50)

    embeddings = analyzer.visualize_features_tsne(features, labels, n_components=3, random_state=42)

    assert embeddings.shape == (50, 3)


# ============================================================================
# Test UMAP Visualization
# ============================================================================


def test_visualize_features_umap(mock_model, random_seed):
    """Test UMAP visualization if available."""
    # Import here to check availability
    try:
        from src.evaluation.feature_quality import UMAP_AVAILABLE

        if not UMAP_AVAILABLE:
            pytest.skip("UMAP not available")
    except ImportError:
        pytest.skip("UMAP not available")

    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    features = np.random.randn(50, 20)
    labels = np.random.randint(0, 5, 50)

    embeddings = analyzer.visualize_features_umap(features, labels, n_components=2, random_state=42)

    assert embeddings.shape == (50, 2)


def test_visualize_features_umap_not_available(mock_model, random_seed):
    """Test that UMAP raises error when not installed."""
    # Mock UMAP as not available
    from src.evaluation import feature_quality

    original_umap = feature_quality.UMAP_AVAILABLE
    feature_quality.UMAP_AVAILABLE = False

    try:
        analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

        features = np.random.randn(50, 20)
        labels = np.random.randint(0, 5, 50)

        with pytest.raises(ImportError, match="UMAP not available"):
            analyzer.visualize_features_umap(features, labels)
    finally:
        feature_quality.UMAP_AVAILABLE = original_umap


# ============================================================================
# Test Complete Analysis
# ============================================================================


def test_compute_all_metrics(mock_model, simple_dataloader, random_seed):
    """Test computing all metrics at once."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    metrics = analyzer.compute_all_metrics(simple_dataloader, max_samples=100)

    assert "rank" in metrics
    assert "statistics" in metrics
    assert "isotropy" in metrics
    assert "collapse" in metrics
    assert "num_samples" in metrics
    assert "feature_dim" in metrics

    # Check nested structure
    assert "effective_rank" in metrics["rank"]
    assert "mean_variance" in metrics["statistics"]
    assert "uniformity" in metrics["isotropy"]
    assert "any_collapse" in metrics["collapse"]


# ============================================================================
# Test Convenience Functions
# ============================================================================


def test_analyze_feature_quality(mock_model, simple_dataloader, random_seed):
    """Test analyze_feature_quality convenience function."""
    metrics = analyze_feature_quality(
        model=mock_model,
        dataloader=simple_dataloader,
        max_samples=100,
        device="cpu",
    )

    assert "rank" in metrics
    assert "statistics" in metrics
    assert "isotropy" in metrics
    assert "collapse" in metrics


def test_print_quality_report(mock_model, simple_dataloader, random_seed, capsys):
    """Test quality report printing."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")
    metrics = analyzer.compute_all_metrics(simple_dataloader, max_samples=50)

    print_quality_report(metrics, verbose=False)

    captured = capsys.readouterr()
    assert "FEATURE QUALITY REPORT" in captured.out
    assert "Effective Rank" in captured.out
    assert "Rank Ratio" in captured.out


def test_compare_hierarchy_levels(mock_model, simple_dataloader, random_seed):
    """Test comparison across hierarchy levels."""
    results = compare_hierarchy_levels(
        model=mock_model,
        dataloader=simple_dataloader,
        num_levels=3,
        max_samples=50,
        device="cpu",
    )

    assert len(results) == 3
    for level in range(3):
        assert level in results
        assert "rank" in results[level]
        assert "statistics" in results[level]


def test_compare_hierarchy_levels_auto_detect(mock_model, simple_dataloader, random_seed):
    """Test hierarchy comparison with auto-detection."""
    # Model has num_hierarchies attribute
    results = compare_hierarchy_levels(
        model=mock_model,
        dataloader=simple_dataloader,
        num_levels=None,  # Auto-detect
        max_samples=50,
        device="cpu",
    )

    # Should use model's num_hierarchies
    assert len(results) == 3


def test_compare_hierarchy_levels_no_attribute(simple_dataloader):
    """Test that error is raised when num_hierarchies not available."""
    # Create mock without num_hierarchies attribute
    mock_no_attr = MagicMock(spec=[])  # Empty spec means no attributes
    mock_no_attr.embed_dim = 384

    def mock_extract(images, level=0, use_target_encoder=True):
        batch_size = images.shape[0]
        return torch.randn(batch_size, 196, 384)

    mock_no_attr.extract_features = mock_extract
    mock_no_attr.eval = MagicMock(return_value=mock_no_attr)
    mock_no_attr.parameters = MagicMock(return_value=[])

    with pytest.raises(ValueError, match="num_hierarchies"):
        compare_hierarchy_levels(
            model=mock_no_attr,
            dataloader=simple_dataloader,
            num_levels=None,
            device="cpu",
        )


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_features_handling(mock_model):
    """Test handling of very small feature sets."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # Very small feature matrix
    features = np.random.randn(5, 10)

    # Should still compute metrics without error
    rank = analyzer.compute_effective_rank(features)
    assert rank >= 1.0


def test_high_dimensional_features(mock_model, random_seed):
    """Test with high-dimensional features."""
    analyzer = FeatureQualityAnalyzer(model=mock_model, device="cpu")

    # More dimensions than samples
    features = np.random.randn(50, 200)

    metrics = analyzer.compute_rank_analysis(features)

    assert metrics["feature_dim"] == 200
    # Effective rank should be at most 50 (number of samples)
    assert metrics["effective_rank"] <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
