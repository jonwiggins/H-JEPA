"""
Shared pytest configuration and fixtures for H-JEPA tests.

This module provides common fixtures and configuration for all test modules.
"""

import pytest
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_device():
    """
    Get the best available device for testing.

    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@pytest.fixture
def device(test_device):
    """Per-test device fixture (alias for test_device)."""
    return test_device


@pytest.fixture
def random_seed():
    """Fix random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def sample_image_224():
    """Create a sample 224x224 PIL image."""
    return Image.new('RGB', (224, 224), color=(128, 128, 128))


@pytest.fixture
def sample_image_96():
    """Create a sample 96x96 PIL image."""
    return Image.new('RGB', (96, 96), color=(100, 100, 100))


@pytest.fixture
def sample_batch_224(device):
    """
    Create a sample batch of 224x224 images and targets.

    Returns:
        tuple: (images, targets)
            - images: [B, C, H, W] tensor
            - targets: [B] tensor with class indices
    """
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 1000, (batch_size,), device=device)
    return images, targets


@pytest.fixture
def sample_batch_small(device):
    """
    Create a small batch (2 samples) for quick tests.
    """
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 100, (batch_size,), device=device)
    return images, targets


@pytest.fixture
def sample_embeddings_2d(device):
    """
    Create sample 2D embeddings for contrastive learning tests.

    Returns:
        tuple: (z_i, z_j) where each is [B, D]
    """
    batch_size = 8
    embed_dim = 128
    z_i = torch.randn(batch_size, embed_dim, device=device)
    z_j = torch.randn(batch_size, embed_dim, device=device)
    return z_i, z_j


@pytest.fixture
def sample_embeddings_3d(device):
    """
    Create sample 3D patch embeddings for tests.

    Returns:
        tuple: (z_i, z_j) where each is [B, N, D]
    """
    batch_size = 4
    num_patches = 196
    embed_dim = 128
    z_i = torch.randn(batch_size, num_patches, embed_dim, device=device)
    z_j = torch.randn(batch_size, num_patches, embed_dim, device=device)
    return z_i, z_j


@pytest.fixture
def tiny_vit_config():
    """Configuration for a tiny ViT model for fast testing."""
    return {
        'encoder_type': 'vit_tiny_patch16_224',
        'img_size': 224,
        'embed_dim': 192,
        'predictor_depth': 2,
        'predictor_num_heads': 3,
        'num_hierarchies': 2,
    }


@pytest.fixture
def small_vit_config():
    """Configuration for a small ViT model for testing."""
    return {
        'encoder_type': 'vit_small_patch16_224',
        'img_size': 224,
        'embed_dim': 384,
        'predictor_depth': 4,
        'predictor_num_heads': 6,
        'num_hierarchies': 3,
    }


@pytest.fixture
def fpn_config():
    """Configuration for FPN testing."""
    return {
        'use_fpn': True,
        'fpn_feature_dim': 192,
        'fpn_fusion_method': 'add',
    }


@pytest.fixture
def training_config():
    """Standard training configuration."""
    return {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 1,
        'use_gradient_checkpointing': False,
    }


@pytest.fixture
def augmentation_config():
    """DeiT III augmentation configuration."""
    return {
        'image_size': 224,
        'auto_augment': True,
        'rand_aug_num_ops': 2,
        'rand_aug_magnitude': 9,
        'random_erasing_prob': 0.25,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'num_classes': 1000,
    }


@pytest.fixture
def contrastive_config():
    """C-JEPA contrastive learning configuration."""
    return {
        'use_contrastive': True,
        'contrastive_weight': 0.1,
        'contrastive_temperature': 0.1,
        'use_cosine_similarity': True,
    }


@pytest.fixture
def multicrop_config():
    """Multi-crop masking configuration."""
    return {
        'global_crop_size': 224,
        'local_crop_size': 96,
        'num_global_crops': 2,
        'num_local_crops': 6,
        'masking_strategy': 'global_only',
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require CUDA"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'integration' marker to integration test classes
        if "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add 'cuda' marker to tests that require CUDA
        if "cuda" in item.nodeid.lower() or "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.cuda)
