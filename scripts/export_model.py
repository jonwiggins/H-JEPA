#!/usr/bin/env python3
"""
Export H-JEPA model to optimized formats for deployment.

Usage:
    python scripts/export_model.py --checkpoint model.pt --output-dir ./exported
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.optimized_model import (
    export_to_onnx,
    export_to_torchscript,
    quantize_model,
)
from src.models.hjepa import create_hjepa

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Export H-JEPA model for deployment')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./exported_models',
        help='Output directory for exported models'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['torchscript', 'onnx', 'quantized'],
        choices=['torchscript', 'onnx', 'quantized'],
        help='Formats to export'
    )
    parser.add_argument(
        '--hierarchy-level',
        type=int,
        default=0,
        help='Hierarchy level for feature extraction'
    )
    parser.add_argument(
        '--quantization-type',
        type=str,
        default='dynamic',
        choices=['dynamic', 'static'],
        help='Type of quantization'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Create model
    logger.info("Creating model...")
    model = create_hjepa(
        encoder_type=model_config.get('encoder_type', 'vit_base_patch16_224'),
        img_size=config.get('data', {}).get('image_size', 224),
        embed_dim=model_config.get('embed_dim', 768),
        predictor_depth=model_config.get('predictor', {}).get('depth', 6),
        predictor_num_heads=model_config.get('predictor', {}).get('num_heads', 12),
        num_hierarchies=model_config.get('num_hierarchies', 3),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"Export formats: {args.formats}")

    # Export to TorchScript
    if 'torchscript' in args.formats:
        logger.info("Exporting to TorchScript...")
        ts_path = os.path.join(args.output_dir, 'model.torchscript.pt')
        try:
            export_to_torchscript(
                model,
                ts_path,
                hierarchy_level=args.hierarchy_level,
                optimize=True,
            )
            logger.info(f"TorchScript model saved to: {ts_path}")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")

    # Export to ONNX
    if 'onnx' in args.formats:
        logger.info("Exporting to ONNX...")
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        try:
            export_to_onnx(
                model,
                onnx_path,
                hierarchy_level=args.hierarchy_level,
                opset_version=14,
                dynamic_axes=True,
            )
            logger.info(f"ONNX model saved to: {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            logger.warning("ONNX export may not be fully supported for all operations")

    # Quantize model
    if 'quantized' in args.formats:
        logger.info(f"Quantizing model ({args.quantization_type})...")
        quant_path = os.path.join(args.output_dir, 'model.quantized.pt')
        try:
            quantize_model(
                model,
                quant_path,
                hierarchy_level=args.hierarchy_level,
                quantization_type=args.quantization_type,
            )
            logger.info(f"Quantized model saved to: {quant_path}")
        except Exception as e:
            logger.error(f"Quantization failed: {e}")

    logger.info("Export complete!")


if __name__ == '__main__':
    main()
