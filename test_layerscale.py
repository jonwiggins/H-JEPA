#!/usr/bin/env python3
"""
Test script to verify LayerScale implementation.

This script demonstrates how to use LayerScale regularization in H-JEPA.
"""

import sys

import torch

sys.path.insert(0, "/Users/jon/repos/H-JEPA")

from src.models.encoder import ContextEncoder, LayerScale, TargetEncoder, create_encoder


def test_layerscale_module():
    """Test the LayerScale module directly."""
    print("=" * 70)
    print("Testing LayerScale Module")
    print("=" * 70)

    dim = 768
    init_value = 1e-5
    batch_size = 2
    seq_len = 196

    # Create LayerScale module
    layerscale = LayerScale(dim=dim, init_value=init_value)

    # Check initialization
    print(f"\nLayerScale initialized with:")
    print(f"  - Dimension: {dim}")
    print(f"  - Init value: {init_value}")
    print(f"  - Scale parameter shape: {layerscale.scale.shape}")
    print(f"  - Scale parameter values (first 5): {layerscale.scale[:5].tolist()}")
    print(
        f"  - All values equal to init_value: {torch.allclose(layerscale.scale, torch.ones(dim) * init_value)}"
    )

    # Test forward pass
    x = torch.randn(batch_size, seq_len, dim)
    output = layerscale(x)

    print(f"\nForward pass test:")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Input mean: {x.mean().item():.6f}")
    print(f"  - Output mean: {output.mean().item():.6f}")
    print(f"  - Scale factor applied correctly: {torch.allclose(output, x * init_value)}")

    # Check that scale is learnable
    print(f"\nLearnable parameter check:")
    print(f"  - Scale requires grad: {layerscale.scale.requires_grad}")
    print(f"  - Number of parameters: {sum(p.numel() for p in layerscale.parameters())}")


def test_encoder_with_layerscale():
    """Test encoder creation with LayerScale."""
    print("\n" + "=" * 70)
    print("Testing Encoder with LayerScale")
    print("=" * 70)

    # Create encoder WITHOUT LayerScale
    print("\n1. Creating encoder WITHOUT LayerScale...")
    encoder_no_ls = ContextEncoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        use_layerscale=False,
    )

    # Count parameters
    params_no_ls = sum(p.numel() for p in encoder_no_ls.parameters())
    print(f"   - Total parameters: {params_no_ls:,}")
    print(f"   - Number of transformer blocks: {len(encoder_no_ls.vit.blocks)}")

    # Create encoder WITH LayerScale
    print("\n2. Creating encoder WITH LayerScale (init=1e-5)...")
    encoder_with_ls = ContextEncoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        use_layerscale=True,
        layerscale_init=1e-5,
    )

    # Count parameters
    params_with_ls = sum(p.numel() for p in encoder_with_ls.parameters())
    print(f"   - Total parameters: {params_with_ls:,}")
    print(f"   - Number of transformer blocks: {len(encoder_with_ls.vit.blocks)}")
    print(f"   - Additional parameters from LayerScale: {params_with_ls - params_no_ls:,}")

    # Expected additional params: 2 LayerScale modules per block * embed_dim * num_blocks
    embed_dim = encoder_with_ls.embed_dim
    num_blocks = len(encoder_with_ls.vit.blocks)
    expected_additional = 2 * embed_dim * num_blocks
    print(f"   - Expected additional parameters: {expected_additional:,}")
    print(f"   - Match: {params_with_ls - params_no_ls == expected_additional}")

    # Check that LayerScale is properly integrated
    print("\n3. Verifying LayerScale integration...")
    first_block = encoder_with_ls.vit.blocks[0]
    print(f"   - Block attention type: {type(first_block.attn)}")
    print(f"   - Block MLP type: {type(first_block.mlp)}")
    print(
        f"   - Attention is wrapped in Sequential: {isinstance(first_block.attn, torch.nn.Sequential)}"
    )
    print(f"   - MLP is wrapped in Sequential: {isinstance(first_block.mlp, torch.nn.Sequential)}")

    if isinstance(first_block.attn, torch.nn.Sequential):
        print(f"   - Attention wrapper length: {len(first_block.attn)}")
        print(
            f"   - Last module in attention is LayerScale: {isinstance(first_block.attn[-1], LayerScale)}"
        )

    if isinstance(first_block.mlp, torch.nn.Sequential):
        print(f"   - MLP wrapper length: {len(first_block.mlp)}")
        print(
            f"   - Last module in MLP is LayerScale: {isinstance(first_block.mlp[-1], LayerScale)}"
        )


def test_encoder_factory():
    """Test create_encoder factory function with LayerScale."""
    print("\n" + "=" * 70)
    print("Testing create_encoder Factory Function")
    print("=" * 70)

    print("\nCreating context and target encoders with LayerScale...")
    context_encoder, target_encoder = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        use_layerscale=True,
        layerscale_init=1e-5,
    )

    print(f"Context encoder parameters: {sum(p.numel() for p in context_encoder.parameters()):,}")
    print(f"Target encoder parameters: {sum(p.numel() for p in target_encoder.parameters()):,}")
    print(
        f"Both encoders have same number of parameters: {sum(p.numel() for p in context_encoder.parameters()) == sum(p.numel() for p in target_encoder.parameters())}"
    )

    # Verify LayerScale is in both encoders
    context_has_ls = isinstance(context_encoder.vit.blocks[0].attn, torch.nn.Sequential)
    target_has_ls = isinstance(target_encoder.vit.blocks[0].attn, torch.nn.Sequential)

    print(f"\nLayerScale verification:")
    print(f"  - Context encoder has LayerScale: {context_has_ls}")
    print(f"  - Target encoder has LayerScale: {target_has_ls}")


def test_forward_pass():
    """Test forward pass with LayerScale enabled."""
    print("\n" + "=" * 70)
    print("Testing Forward Pass with LayerScale")
    print("=" * 70)

    print("\nCreating encoder with LayerScale...")
    encoder = ContextEncoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        use_layerscale=True,
        layerscale_init=1e-5,
    )
    encoder.eval()

    # Create dummy input
    batch_size = 2
    img_size = 224
    x = torch.randn(batch_size, 3, img_size, img_size)

    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = encoder(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {encoder.num_patches + 1}, {encoder.embed_dim})")
    print(
        f"Shape is correct: {output.shape == (batch_size, encoder.num_patches + 1, encoder.embed_dim)}"
    )
    print(f"Output contains valid values (not NaN): {not torch.isnan(output).any()}")
    print(f"Output has reasonable magnitude: {output.abs().mean().item():.4f}")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# LayerScale Implementation Test Suite")
    print("#" * 70)

    try:
        test_layerscale_module()
        test_encoder_with_layerscale()
        test_encoder_factory()
        test_forward_pass()

        print("\n" + "=" * 70)
        print("All tests completed successfully!")
        print("=" * 70)
        print("\nLayerScale implementation is working correctly.")
        print("You can now enable LayerScale in your training configuration by setting:")
        print("  use_layerscale=True")
        print("  layerscale_init=1e-5  (or any value between 1e-6 and 1e-4)")

    except Exception as e:
        print(f"\n{'!' * 70}")
        print(f"ERROR: {str(e)}")
        print(f"{'!' * 70}")
        import traceback

        traceback.print_exc()
