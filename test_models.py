"""
Quick test script to verify H-JEPA model implementation.
"""

import torch
import yaml
from src.models import HJEPA, create_hjepa, create_hjepa_from_config


def test_basic_model():
    """Test basic model creation and forward pass."""
    print("Testing basic H-JEPA model...")

    # Create model
    model = create_hjepa(
        encoder_type="vit_small_patch16_224",  # Use small for faster testing
        img_size=224,
        embed_dim=384,
        predictor_depth=4,
        predictor_num_heads=6,
        num_hierarchies=3,
    )

    print(f"✓ Model created successfully")
    print(f"  - Embed dim: {model.embed_dim}")
    print(f"  - Num hierarchies: {model.num_hierarchies}")
    print(f"  - Num patches: {model.get_num_patches()}")
    print(f"  - Patch size: {model.get_patch_size()}")

    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    # Create dummy mask (50% masked)
    num_patches = model.get_num_patches()
    mask = torch.zeros(batch_size, num_patches)
    num_masked = num_patches // 2
    mask[:, :num_masked] = 1  # Mask first half

    print(f"\n✓ Created dummy inputs")
    print(f"  - Image shape: {images.shape}")
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Num masked patches: {num_masked}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images, mask, return_all_levels=True)

    print(f"\n✓ Forward pass successful")
    print(f"  - Num hierarchy levels: {len(outputs['predictions'])}")

    for i, (pred, target) in enumerate(zip(outputs['predictions'], outputs['targets'])):
        print(f"  - Level {i}: pred shape {pred.shape}, target shape {target.shape}")

    # Test feature extraction
    features = model.extract_features(images, level=0)
    print(f"\n✓ Feature extraction successful")
    print(f"  - Features shape: {features.shape}")

    # Test EMA update
    momentum = model.update_target_encoder(current_step=100)
    print(f"\n✓ EMA update successful")
    print(f"  - Current momentum: {momentum:.4f}")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


def test_config_loading():
    """Test model creation from config."""
    print("\n\nTesting config-based model creation...")

    # Load config
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Modify for testing (use smaller model)
    config['model']['encoder_type'] = 'vit_small_patch16_224'
    config['model']['embed_dim'] = 384

    # Create model from config
    model = create_hjepa_from_config(config)

    print(f"✓ Model created from config")
    print(f"  - Encoder type: {config['model']['encoder_type']}")
    print(f"  - Embed dim: {model.embed_dim}")
    print(f"  - Num hierarchies: {model.num_hierarchies}")

    print("\n" + "=" * 50)
    print("Config test passed! ✓")
    print("=" * 50)


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_parameter_count():
    """Test parameter counting."""
    print("\n\nTesting parameter counts...")

    model = create_hjepa(
        encoder_type="vit_small_patch16_224",
        embed_dim=384,
    )

    total, trainable = count_parameters(model)

    print(f"✓ Parameter counts:")
    print(f"  - Total parameters: {total:,}")
    print(f"  - Trainable parameters: {trainable:,}")
    print(f"  - Non-trainable (EMA): {total - trainable:,}")

    print("\n" + "=" * 50)
    print("Parameter count test passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("H-JEPA Model Tests")
    print("=" * 50)

    try:
        test_basic_model()
        test_config_loading()
        test_parameter_count()

        print("\n\n" + "=" * 50)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
        print("=" * 50)

    except Exception as e:
        print(f"\n\n✗ Test failed with error:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
