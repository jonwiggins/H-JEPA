"""
Optimized inference implementations for H-JEPA.

Provides:
- TorchScript export
- ONNX export
- INT8 quantization
- Batch inference utilities
- Optimized feature extraction
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..models.hjepa import HJEPA

logger = logging.getLogger(__name__)


class OptimizedHJEPA(nn.Module):
    """
    Optimized H-JEPA model for inference.

    Removes training-only components and optimizes for feature extraction.
    """

    def __init__(self, hjepa_model: HJEPA, hierarchy_level: int = 0) -> None:
        """
        Initialize optimized model.

        Args:
            hjepa_model: Trained H-JEPA model
            hierarchy_level: Hierarchy level to use for feature extraction
        """
        super().__init__()

        self.hierarchy_level = hierarchy_level
        self.embed_dim = hjepa_model.embed_dim
        self.num_hierarchies = hjepa_model.num_hierarchies

        # Use only target encoder for inference
        self.encoder = hjepa_model.target_encoder

        # Copy hierarchy projection and pooling
        self.hierarchy_projection = hjepa_model.hierarchy_projections[hierarchy_level]
        self.hierarchy_pooling = hjepa_model.hierarchy_pooling[hierarchy_level]

        # Set to eval mode
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Features [B, N, D] or [B, N', D] depending on hierarchy level
        """
        # Encode image
        features = self.encoder(images)

        # Exclude CLS token
        features = features[:, 1:, :]

        # Project to hierarchy level
        features = self.hierarchy_projection(features)

        # Apply pooling for coarser levels
        if self.hierarchy_level > 0:
            B, N, D = features.shape
            features = features.transpose(1, 2)  # [B, D, N]
            features = self.hierarchy_pooling(features)
            features = features.transpose(1, 2)  # [B, N', D]

        return features


def export_to_torchscript(
    model: Union[HJEPA, OptimizedHJEPA],
    output_path: str,
    example_input: Optional[torch.Tensor] = None,
    hierarchy_level: int = 0,
    optimize: bool = True,
) -> torch.jit.ScriptModule:
    """
    Export model to TorchScript.

    Args:
        model: H-JEPA model to export
        output_path: Path to save TorchScript model
        example_input: Example input tensor for tracing
        hierarchy_level: Hierarchy level (only used if model is HJEPA)
        optimize: Whether to optimize for inference

    Returns:
        TorchScript module
    """
    logger.info("Exporting model to TorchScript...")

    # Create optimized model if needed
    if isinstance(model, HJEPA):
        model = OptimizedHJEPA(model, hierarchy_level=hierarchy_level)

    model.eval()

    # Create example input if not provided
    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224)

    # Move to same device as model
    device = next(model.parameters()).device
    example_input = example_input.to(device)

    try:
        # Trace model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)

        # Optimize for inference
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)

        # Save
        torch.jit.save(traced_model, output_path)
        logger.info(f"TorchScript model saved to: {output_path}")

        return traced_model

    except Exception as e:
        logger.error(f"Failed to export to TorchScript: {e}")
        raise


def export_to_onnx(
    model: Union[HJEPA, OptimizedHJEPA],
    output_path: str,
    example_input: Optional[torch.Tensor] = None,
    hierarchy_level: int = 0,
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> None:
    """
    Export model to ONNX format.

    Args:
        model: H-JEPA model to export
        output_path: Path to save ONNX model
        example_input: Example input tensor
        hierarchy_level: Hierarchy level (only used if model is HJEPA)
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic batch size
    """
    logger.info("Exporting model to ONNX...")

    # Create optimized model if needed
    if isinstance(model, HJEPA):
        model = OptimizedHJEPA(model, hierarchy_level=hierarchy_level)

    model.eval()

    # Create example input if not provided
    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224)

    # Move to same device as model
    device = next(model.parameters()).device
    example_input = example_input.to(device)

    # Set up dynamic axes
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "images": {0: "batch_size"},
            "features": {0: "batch_size"},
        }

    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=["images"],
            output_names=["features"],
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )

        logger.info(f"ONNX model saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to export to ONNX: {e}")
        logger.warning("ONNX export may not be fully supported for all operations")
        raise


def quantize_model(
    model: Union[HJEPA, OptimizedHJEPA],
    output_path: str,
    hierarchy_level: int = 0,
    quantization_type: str = "dynamic",
    calibration_data: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Quantize model to INT8 for faster inference.

    Args:
        model: H-JEPA model to quantize
        output_path: Path to save quantized model
        hierarchy_level: Hierarchy level (only used if model is HJEPA)
        quantization_type: Type of quantization ('dynamic' or 'static')
        calibration_data: Calibration data for static quantization

    Returns:
        Quantized model
    """
    logger.info(f"Quantizing model using {quantization_type} quantization...")

    # Create optimized model if needed
    if isinstance(model, HJEPA):
        model = OptimizedHJEPA(model, hierarchy_level=hierarchy_level)

    model.eval()
    model.cpu()  # Quantization requires CPU

    try:
        if quantization_type == "dynamic":
            # Dynamic quantization (weights only)
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

        elif quantization_type == "static":
            # Static quantization (requires calibration)
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration data")

            # Prepare model for quantization
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(model, inplace=True)

            # Calibrate
            with torch.no_grad():
                model(calibration_data)

            # Convert
            quantized_model = torch.quantization.convert(model, inplace=False)

        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        logger.info(f"Quantized model saved to: {output_path}")

        return quantized_model

    except Exception as e:
        logger.error(f"Failed to quantize model: {e}")
        raise


class BatchInference:
    """
    Utility for efficient batch inference.

    Handles batching, preprocessing, and feature extraction.
    """

    def __init__(
        self,
        model: Union[HJEPA, OptimizedHJEPA, torch.jit.ScriptModule],
        device: str = "cuda",
        batch_size: int = 32,
        hierarchy_level: int = 0,
    ) -> None:
        """
        Initialize batch inference.

        Args:
            model: Model for inference
            device: Device to use
            batch_size: Batch size for inference
            hierarchy_level: Hierarchy level for feature extraction
        """
        self.device = device
        self.batch_size = batch_size
        self.hierarchy_level = hierarchy_level

        # Create optimized model if needed
        if isinstance(model, HJEPA):
            model = OptimizedHJEPA(model, hierarchy_level=hierarchy_level)

        self.model = model.to(device).eval()

    @torch.no_grad()
    def extract_features(
        self,
        images: Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
        return_numpy: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract features from images in batches.

        Args:
            images: Input images as tensor, numpy array, or list
            return_numpy: Whether to return numpy array

        Returns:
            Extracted features
        """
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        elif isinstance(images, list):
            images = torch.stack(images)

        images = images.to(self.device)

        # Process in batches
        all_features = []
        num_images = images.shape[0]

        for i in range(0, num_images, self.batch_size):
            batch = images[i : i + self.batch_size]
            features = self.model(batch)
            all_features.append(features)

        # Concatenate
        all_features = torch.cat(all_features, dim=0)

        if return_numpy:
            return all_features.cpu().numpy()
        return all_features

    def benchmark(
        self,
        num_images: int = 100,
        num_runs: int = 10,
        warmup_runs: int = 5,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            num_images: Number of images to process
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Create dummy data
        dummy_images = torch.randn(num_images, 3, 224, 224).to(self.device)

        # Warmup
        logger.info(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            _ = self.extract_features(dummy_images, return_numpy=False)

        # Synchronize for accurate timing
        if self.device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        logger.info(f"Running {num_runs} benchmark iterations...")
        times = []

        for _ in range(num_runs):
            start_time = time.time()
            _ = self.extract_features(dummy_images, return_numpy=False)

            if self.device == "cuda":
                torch.cuda.synchronize()

            times.append(time.time() - start_time)

        # Calculate statistics
        times = np.array(times)
        results = {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "throughput_images_per_sec": num_images / np.mean(times),
            "latency_per_image_ms": (np.mean(times) / num_images) * 1000,
        }

        logger.info(f"Benchmark results: {results}")
        return results


def create_inference_config(
    checkpoint_path: str,
    export_formats: List[str] = ["torchscript", "onnx"],
    quantize: bool = True,
    hierarchy_level: int = 0,
    output_dir: str = "./exported_models",
) -> Dict[str, str]:
    """
    Create optimized inference models from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        export_formats: Formats to export ('torchscript', 'onnx')
        quantize: Whether to create quantized version
        hierarchy_level: Hierarchy level for feature extraction
        output_dir: Directory to save exported models

    Returns:
        Dictionary mapping format to output path
    """
    import os

    from ..models.hjepa import create_hjepa
    from ..utils.checkpoint import load_checkpoint

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    # Create model
    model = create_hjepa(**config.get("model", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    exported_paths = {}

    # Export to TorchScript
    if "torchscript" in export_formats:
        ts_path = os.path.join(output_dir, "model.torchscript.pt")
        export_to_torchscript(model, ts_path, hierarchy_level=hierarchy_level)
        exported_paths["torchscript"] = ts_path

    # Export to ONNX
    if "onnx" in export_formats:
        onnx_path = os.path.join(output_dir, "model.onnx")
        try:
            export_to_onnx(model, onnx_path, hierarchy_level=hierarchy_level)
            exported_paths["onnx"] = onnx_path
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    # Quantize model
    if quantize:
        quant_path = os.path.join(output_dir, "model.quantized.pt")
        quantize_model(model, quant_path, hierarchy_level=hierarchy_level)
        exported_paths["quantized"] = quant_path

    logger.info(f"Exported models: {exported_paths}")
    return exported_paths
