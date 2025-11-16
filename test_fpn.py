"""
Test script to verify FPN implementation in H-JEPA.

This script tests:
1. Model creation with FPN enabled
2. Forward pass with FPN
3. Feature extraction with FPN
4. Both fusion methods ('add' and 'concat')
"""

import torch
from src.models.hjepa import create_hjepa


def test_fpn_creation():
    """Test that FPN models can be created successfully."""
    print("Testing FPN model creation...")

    # Test with 'add' fusion
    model_add = create_hjepa(
        encoder_type='vit_small_patch16_224',
        img_size=224,
        embed_dim=384,
        num_hierarchies=3,
        use_fpn=True,
        fpn_feature_dim=256,
        fpn_fusion_method='add',
    )
    print(f"  Created model with 'add' fusion: {type(model_add).__name__}")

    # Test with 'concat' fusion
    model_concat = create_hjepa(
        encoder_type='vit_small_patch16_224',
        img_size=224,
        embed_dim=384,
        num_hierarchies=3,
        use_fpn=True,
        fpn_feature_dim=256,
        fpn_fusion_method='concat',
    )
    print(f"  Created model with 'concat' fusion: {type(model_concat).__name__}")

    # Test without FPN for comparison
    model_no_fpn = create_hjepa(
        encoder_type='vit_small_patch16_224',
        img_size=224,
        embed_dim=384,
        num_hierarchies=3,
        use_fpn=False,
    )
    print(f"  Created model without FPN: {type(model_no_fpn).__name__}")

    print("  ✓ Model creation successful\n")
    return model_add, model_concat, model_no_fpn


def test_fpn_forward_pass(model, fusion_method='add'):
    """Test forward pass through FPN model."""
    print(f"Testing forward pass with '{fusion_method}' fusion...")

    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    # Create dummy mask (50% of patches masked)
    num_patches = model.get_num_patches()
    mask = torch.zeros(batch_size, num_patches)
    num_masked = num_patches // 2
    mask[:, :num_masked] = 1

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images, mask, return_all_levels=True)

    # Check outputs
    predictions = outputs['predictions']
    targets = outputs['targets']

    print(f"  Number of hierarchy levels: {len(predictions)}")
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"  Level {i}: pred shape={pred.shape}, target shape={target.shape}")

    print(f"  ✓ Forward pass successful\n")
    return outputs


def test_fpn_feature_extraction(model, fusion_method='add'):
    """Test feature extraction at different hierarchy levels."""
    print(f"Testing feature extraction with '{fusion_method}' fusion...")

    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        for level in range(model.num_hierarchies):
            features = model.extract_features(images, level=level)
            print(f"  Level {level}: features shape={features.shape}")

    print(f"  ✓ Feature extraction successful\n")


def compare_parameter_counts():
    """Compare parameter counts for different configurations."""
    print("Comparing parameter counts...")

    configs = [
        {'use_fpn': False, 'fpn_fusion_method': 'add', 'name': 'No FPN'},
        {'use_fpn': True, 'fpn_feature_dim': 256, 'fpn_fusion_method': 'add', 'name': 'FPN (add, dim=256)'},
        {'use_fpn': True, 'fpn_feature_dim': None, 'fpn_fusion_method': 'add', 'name': 'FPN (add, dim=embed)'},
        {'use_fpn': True, 'fpn_feature_dim': 256, 'fpn_fusion_method': 'concat', 'name': 'FPN (concat, dim=256)'},
    ]

    for config in configs:
        name = config.pop('name')
        model = create_hjepa(
            encoder_type='vit_small_patch16_224',
            img_size=224,
            embed_dim=384,
            num_hierarchies=3,
            **config
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  {name}:")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")

    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("FPN Implementation Test Suite")
    print("=" * 60)
    print()

    # Test 1: Model creation
    model_add, model_concat, model_no_fpn = test_fpn_creation()

    # Test 2: Forward pass with 'add' fusion
    test_fpn_forward_pass(model_add, fusion_method='add')

    # Test 3: Forward pass with 'concat' fusion
    test_fpn_forward_pass(model_concat, fusion_method='concat')

    # Test 4: Forward pass without FPN
    test_fpn_forward_pass(model_no_fpn, fusion_method='none')

    # Test 5: Feature extraction with FPN
    test_fpn_feature_extraction(model_add, fusion_method='add')
    test_fpn_feature_extraction(model_concat, fusion_method='concat')

    # Test 6: Parameter count comparison
    compare_parameter_counts()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
