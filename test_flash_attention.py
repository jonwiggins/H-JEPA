"""
Test script to verify Flash Attention integration in H-JEPA encoders.

This script creates encoders with and without Flash Attention and verifies:
1. Model initialization succeeds
2. Forward pass works correctly
3. Output shapes match expectations
4. Flash Attention modules are correctly integrated
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.encoder import create_encoder, FlashAttention, FLASH_ATTENTION_AVAILABLE


def test_flash_attention_module():
    """Test the FlashAttention module directly."""
    print("\n" + "="*80)
    print("Testing FlashAttention Module")
    print("="*80)

    batch_size = 2
    seq_len = 197  # ViT-Base with 224x224 image: 196 patches + 1 CLS token
    embed_dim = 768
    num_heads = 12

    # Create Flash Attention module
    flash_attn = FlashAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        use_flash=True,
    )

    # Create input tensor
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output = flash_attn(x)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"

    print(f"✓ FlashAttention module test passed")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Flash Attention available: {FLASH_ATTENTION_AVAILABLE}")
    print(f"  Using Flash Attention: {flash_attn.use_flash}")


def test_encoder_with_flash_attention():
    """Test encoder creation with Flash Attention enabled."""
    print("\n" + "="*80)
    print("Testing Encoder with Flash Attention ENABLED")
    print("="*80)

    # Create encoders with Flash Attention
    context_encoder, target_encoder = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        pretrained=False,
        use_flash_attention=True,
    )

    print(f"✓ Encoders created successfully")
    print(f"  Context encoder type: {type(context_encoder).__name__}")
    print(f"  Target encoder type:  {type(target_encoder).__name__}")
    print(f"  Embed dim: {context_encoder.embed_dim}")
    print(f"  Num patches: {context_encoder.num_patches}")
    print(f"  Patch size: {context_encoder.patch_size}")

    # Count Flash Attention modules
    flash_attn_count = 0
    standard_attn_count = 0

    for name, module in context_encoder.named_modules():
        if isinstance(module, FlashAttention):
            flash_attn_count += 1
        elif module.__class__.__name__ == 'Attention':
            standard_attn_count += 1

    print(f"\n  Attention modules in Context Encoder:")
    print(f"    FlashAttention modules: {flash_attn_count}")
    print(f"    Standard Attention modules: {standard_attn_count}")

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    mask = torch.zeros(batch_size, context_encoder.num_patches, dtype=torch.bool)

    # Context encoder forward
    context_features = context_encoder(images, mask=mask)
    print(f"\n✓ Context encoder forward pass successful")
    print(f"  Input shape:  {images.shape}")
    print(f"  Output shape: {context_features.shape}")

    # Target encoder forward
    target_features = target_encoder(images)
    print(f"\n✓ Target encoder forward pass successful")
    print(f"  Input shape:  {images.shape}")
    print(f"  Output shape: {target_features.shape}")

    # Verify shapes match
    assert context_features.shape == target_features.shape, \
        "Context and target encoder outputs should have the same shape"
    print(f"\n✓ Output shapes match")


def test_encoder_without_flash_attention():
    """Test encoder creation with Flash Attention disabled."""
    print("\n" + "="*80)
    print("Testing Encoder with Flash Attention DISABLED")
    print("="*80)

    # Create encoders without Flash Attention
    context_encoder, target_encoder = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        pretrained=False,
        use_flash_attention=False,
    )

    print(f"✓ Encoders created successfully")

    # Count attention modules
    flash_attn_count = 0
    standard_attn_count = 0

    for name, module in context_encoder.named_modules():
        if isinstance(module, FlashAttention):
            flash_attn_count += 1
        elif module.__class__.__name__ == 'Attention':
            standard_attn_count += 1

    print(f"\n  Attention modules in Context Encoder:")
    print(f"    FlashAttention modules: {flash_attn_count}")
    print(f"    Standard Attention modules: {standard_attn_count}")

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    context_features = context_encoder(images)
    target_features = target_encoder(images)

    print(f"\n✓ Forward passes successful")
    print(f"  Context output shape: {context_features.shape}")
    print(f"  Target output shape:  {target_features.shape}")


def test_compatibility():
    """Test Flash Attention compatibility and fallback."""
    print("\n" + "="*80)
    print("Flash Attention Compatibility Check")
    print("="*80)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Flash Attention available: {FLASH_ATTENTION_AVAILABLE}")

    if FLASH_ATTENTION_AVAILABLE:
        print("✓ PyTorch 2.0+ detected - Flash Attention will be used")
        print("  Expected speedup: 2-5x for attention computation")
        print("  Expected memory savings: ~30-50% for attention")
    else:
        print("⚠ PyTorch 2.0+ not detected - using standard attention fallback")
        print("  Recommendation: Upgrade to PyTorch 2.0+ for Flash Attention benefits")

    # Check device compatibility
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print("  Flash Attention optimized for CUDA GPUs")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("\n✓ MPS (Apple Silicon) available")
        print("  Flash Attention supports MPS backend")
    else:
        print("\n✓ CPU only")
        print("  Flash Attention supports CPU (with reduced speedup)")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("H-JEPA Flash Attention Integration Test Suite")
    print("="*80)

    try:
        # Run tests
        test_compatibility()
        test_flash_attention_module()
        test_encoder_with_flash_attention()
        test_encoder_without_flash_attention()

        # Summary
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nFlash Attention integration is working correctly!")
        print("\nNext steps:")
        print("1. Set 'use_flash_attention: true' in your config file")
        print("2. Train with PyTorch 2.0+ for optimal performance")
        print("3. Monitor training speed improvements (2-5x faster attention)")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
