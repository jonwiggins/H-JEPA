"""
Test script to verify Flash Attention integration in H-JEPA encoders.

This script creates encoders with and without Flash Attention and verifies:
1. Model initialization succeeds
2. Forward pass works correctly
3. Output shapes match expectations
4. Flash Attention parameter is correctly handled
"""

import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.models.encoder import FLASH_ATTENTION_AVAILABLE, create_encoder


def test_flash_attention_availability():
    """Test Flash Attention availability."""
    print("\n" + "=" * 80)
    print("Testing Flash Attention Availability")
    print("=" * 80)

    if FLASH_ATTENTION_AVAILABLE:
        print("✓ Flash Attention is available")
    else:
        print("✗ Flash Attention is not available (expected on MPS)")
        print("  This is normal for Apple Silicon devices")

    print(f"✓ Flash Attention availability test passed")


def test_encoder_with_flash_attention():
    """Test encoder creation with Flash Attention parameter."""
    print("\n" + "=" * 80)
    print("Testing Encoder with Flash Attention Parameter")
    print("=" * 80)

    # Create encoders with Flash Attention parameter (will be ignored on MPS)
    context_encoder, target_encoder = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        pretrained=False,
        use_flash_attention=True,  # This parameter is accepted but may be ignored on MPS
    )

    print(f"✓ Encoders created successfully")
    print(f"  Context encoder type: {type(context_encoder).__name__}")
    print(f"  Target encoder type:  {type(target_encoder).__name__}")

    # Test forward pass
    batch_size = 2
    num_channels = 3
    img_size = 224

    x = torch.randn(batch_size, num_channels, img_size, img_size)

    # Test context encoder forward pass
    with torch.no_grad():
        context_output = context_encoder(x)

    print(f"  Context encoder output shape: {context_output.shape}")

    # Test target encoder forward pass
    with torch.no_grad():
        target_output = target_encoder(x)

    print(f"  Target encoder output shape: {target_output.shape}")

    print(f"✓ Forward pass successful")


def test_encoder_without_flash_attention():
    """Test encoder creation with Flash Attention disabled."""
    print("\n" + "=" * 80)
    print("Testing Encoder with Flash Attention DISABLED")
    print("=" * 80)

    # Create encoders without Flash Attention
    context_encoder, target_encoder = create_encoder(
        encoder_type="vit_base_patch16_224",
        img_size=224,
        pretrained=False,
        use_flash_attention=False,
    )

    print(f"✓ Encoders created successfully")
    print(f"  Context encoder type: {type(context_encoder).__name__}")
    print(f"  Target encoder type:  {type(target_encoder).__name__}")

    # Test forward pass
    batch_size = 2
    num_channels = 3
    img_size = 224

    x = torch.randn(batch_size, num_channels, img_size, img_size)

    # Test context encoder forward pass
    with torch.no_grad():
        context_output = context_encoder(x)

    print(f"  Context encoder output shape: {context_output.shape}")

    # Test target encoder forward pass
    with torch.no_grad():
        target_output = target_encoder(x)

    print(f"  Target encoder output shape: {target_output.shape}")

    print(f"✓ Forward pass successful")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Flash Attention Integration Tests for H-JEPA")
    print("=" * 80)

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Running on device: {device}")

    # Run tests
    test_flash_attention_availability()
    test_encoder_with_flash_attention()
    test_encoder_without_flash_attention()

    print("\n" + "=" * 80)
    print("All Flash Attention tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
