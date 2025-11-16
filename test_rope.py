"""
Test script for Rotary Position Embeddings (RoPE) implementation.

This script verifies that:
1. RoPE module can be instantiated
2. Encoders can be created with and without RoPE
3. RoPE correctly rotates Q and K embeddings
4. Forward pass works with RoPE enabled
5. RoPE maintains backward compatibility
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.encoder import VisionRoPE2D, RoPEAttentionWrapper, create_encoder


def test_rope_module():
    """Test VisionRoPE2D module."""
    print("\n" + "="*60)
    print("Testing VisionRoPE2D Module")
    print("="*60)
    
    # Create RoPE module
    dim = 64  # head dimension
    patch_size = 16
    num_patches_per_side = 14  # For 224x224 with 16x16 patches
    
    rope = VisionRoPE2D(
        dim=dim,
        patch_size=patch_size,
        num_patches_per_side=num_patches_per_side,
        theta=10000.0,
    )
    
    print(f"✓ RoPE module created with:")
    print(f"  - Dimension: {dim}")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Grid size: {num_patches_per_side}x{num_patches_per_side}")
    print(f"  - Theta: 10000.0")
    
    # Test forward pass
    batch_size = 2
    num_heads = 12
    seq_len = num_patches_per_side ** 2  # 196 patches
    
    q = torch.randn(batch_size, num_heads, seq_len, dim)
    k = torch.randn(batch_size, num_heads, seq_len, dim)
    
    q_rotated, k_rotated = rope(q, k)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input Q shape: {q.shape}")
    print(f"  - Output Q shape: {q_rotated.shape}")
    print(f"  - Input K shape: {k.shape}")
    print(f"  - Output K shape: {k_rotated.shape}")
    
    # Verify shapes match
    assert q_rotated.shape == q.shape, "Q shape mismatch"
    assert k_rotated.shape == k.shape, "K shape mismatch"
    print("✓ Shape verification passed")
    
    # Verify rotation changes the values
    assert not torch.allclose(q, q_rotated), "Q should be rotated"
    assert not torch.allclose(k, k_rotated), "K should be rotated"
    print("✓ Rotation applied successfully")
    
    # Test dynamic resolution
    new_grid_size = 16
    q_large = torch.randn(batch_size, num_heads, new_grid_size**2, dim)
    k_large = torch.randn(batch_size, num_heads, new_grid_size**2, dim)
    
    q_rot_large, k_rot_large = rope(q_large, k_large, 
                                      num_patches_h=new_grid_size,
                                      num_patches_w=new_grid_size)
    
    print(f"\n✓ Dynamic resolution support:")
    print(f"  - New grid size: {new_grid_size}x{new_grid_size}")
    print(f"  - Output Q shape: {q_rot_large.shape}")
    print(f"  - Output K shape: {k_rot_large.shape}")
    
    return True


def test_encoder_without_rope():
    """Test encoder creation without RoPE (backward compatibility)."""
    print("\n" + "="*60)
    print("Testing Encoder WITHOUT RoPE (Backward Compatibility)")
    print("="*60)
    
    context_encoder, target_encoder = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        pretrained=False,
        use_rope=False,
    )
    
    print("✓ Encoders created without RoPE")
    print(f"  - Context encoder embed_dim: {context_encoder.embed_dim}")
    print(f"  - Target encoder embed_dim: {target_encoder.embed_dim}")
    print(f"  - Number of patches: {context_encoder.num_patches}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        context_out = context_encoder(x)
        target_out = target_encoder(x)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Context output shape: {context_out.shape}")
    print(f"  - Target output shape: {target_out.shape}")
    
    # Expected shape: [batch, num_patches + 1 (cls), embed_dim]
    expected_seq_len = context_encoder.num_patches + 1
    assert context_out.shape == (batch_size, expected_seq_len, context_encoder.embed_dim)
    print("✓ Output shape verification passed")
    
    return True


def test_encoder_with_rope():
    """Test encoder creation with RoPE enabled."""
    print("\n" + "="*60)
    print("Testing Encoder WITH RoPE Enabled")
    print("="*60)
    
    context_encoder, target_encoder = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        pretrained=False,
        use_rope=True,
        rope_theta=10000.0,
    )
    
    print("✓ Encoders created with RoPE")
    print(f"  - Context encoder use_rope: {context_encoder.use_rope}")
    print(f"  - Target encoder use_rope: {target_encoder.use_rope}")
    print(f"  - RoPE theta: 10000.0")
    
    # Verify RoPE module exists
    assert hasattr(context_encoder, 'rope'), "Context encoder should have rope module"
    assert hasattr(target_encoder, 'rope'), "Target encoder should have rope module"
    print("✓ RoPE modules attached to encoders")
    
    # Verify attention layers are wrapped
    first_block_attn = context_encoder.vit.blocks[0].attn
    assert isinstance(first_block_attn, RoPEAttentionWrapper), \
        "Attention should be wrapped with RoPEAttentionWrapper"
    print("✓ Attention layers wrapped with RoPE")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        context_out = context_encoder(x)
        target_out = target_encoder(x)
    
    print(f"\n✓ Forward pass successful:")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Context output shape: {context_out.shape}")
    print(f"  - Target output shape: {target_out.shape}")
    
    # Test with masking
    num_patches = context_encoder.num_patches
    mask = torch.randint(0, 2, (batch_size, num_patches), dtype=torch.bool)
    
    with torch.no_grad():
        masked_out = context_encoder(x, mask=mask)
    
    print(f"\n✓ Forward pass with masking successful:")
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Masked output shape: {masked_out.shape}")
    
    return True


def test_ema_update_with_rope():
    """Test EMA update works correctly with RoPE."""
    print("\n" + "="*60)
    print("Testing EMA Update with RoPE")
    print("="*60)
    
    context_encoder, target_encoder = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        pretrained=False,
        use_rope=True,
    )
    
    # Store initial target weights
    initial_weight = target_encoder.vit.blocks[0].attn.attn.qkv.weight.clone()
    
    # Modify context encoder
    with torch.no_grad():
        context_encoder.vit.blocks[0].attn.attn.qkv.weight.add_(0.1)
    
    # Perform EMA update
    momentum = target_encoder.update_from_context_encoder(context_encoder, current_step=500)
    
    # Verify weights changed
    updated_weight = target_encoder.vit.blocks[0].attn.attn.qkv.weight
    assert not torch.allclose(initial_weight, updated_weight), "EMA should update weights"
    
    print(f"✓ EMA update successful:")
    print(f"  - Momentum: {momentum:.4f}")
    print(f"  - Weight change: {(updated_weight - initial_weight).abs().max():.6f}")
    
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through RoPE."""
    print("\n" + "="*60)
    print("Testing Gradient Flow Through RoPE")
    print("="*60)
    
    context_encoder, _ = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        pretrained=False,
        use_rope=True,
    )
    
    # Enable training mode
    context_encoder.train()
    
    # Create input with gradient tracking
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    
    # Forward pass
    output = context_encoder(x)
    
    # Compute loss and backward
    loss = output.mean()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Input gradients should exist"
    assert context_encoder.vit.blocks[0].attn.attn.qkv.weight.grad is not None, \
        "Attention weights should have gradients"
    
    print("✓ Gradient flow verified:")
    print(f"  - Input grad shape: {x.grad.shape}")
    print(f"  - Input grad norm: {x.grad.norm():.6f}")
    print(f"  - QKV weight grad norm: {context_encoder.vit.blocks[0].attn.attn.qkv.weight.grad.norm():.6f}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ROTARY POSITION EMBEDDINGS (RoPE) TEST SUITE")
    print("="*70)
    
    tests = [
        ("RoPE Module", test_rope_module),
        ("Encoder without RoPE", test_encoder_without_rope),
        ("Encoder with RoPE", test_encoder_with_rope),
        ("EMA Update with RoPE", test_ema_update_with_rope),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n{'✓'*30}")
            print(f"✓ {test_name}: PASSED")
            print(f"{'✓'*30}")
        except Exception as e:
            failed += 1
            print(f"\n{'✗'*30}")
            print(f"✗ {test_name}: FAILED")
            print(f"Error: {str(e)}")
            print(f"{'✗'*30}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 All tests passed! RoPE implementation is working correctly.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
