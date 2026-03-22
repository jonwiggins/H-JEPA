"""
Device management utilities for H-JEPA.

Centralizes device detection, capability queries, and device-specific
operations that were previously scattered across encoder, trainer, and losses.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Centralizes device-specific logic for training and inference.

    Provides capability queries and device-specific operations so that
    model/trainer code can avoid direct device-type branching.

    Args:
        device: Device string ('cuda', 'mps', 'cpu') or torch.device
    """

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device) if isinstance(device, str) else device

    @property
    def type(self) -> str:
        return self.device.type

    @property
    def supports_amp(self) -> bool:
        """Whether automatic mixed precision is supported."""
        return self.device.type == "cuda"

    @property
    def supports_flash_attention(self) -> bool:
        """Whether Flash Attention (scaled_dot_product_attention) is available."""
        if self.device.type != "cuda":
            return False
        return hasattr(torch.nn.functional, "scaled_dot_product_attention")

    @property
    def supports_svd(self) -> bool:
        """Whether torch.svd works on this device (MPS does not support it)."""
        return self.device.type != "mps"

    @property
    def autocast_device_type(self) -> str:
        """Device type string for torch.amp.autocast. MPS falls back to cpu."""
        return self.device.type if self.device.type != "mps" else "cpu"

    def empty_cache(self) -> None:
        """Clear device memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            gc.collect()

    def memory_stats(self) -> dict[str, float]:
        """Return current memory usage in GB."""
        stats: dict[str, float] = {}
        if self.device.type == "cuda":
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        elif self.device.type == "mps":
            stats["allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
            stats["driver_gb"] = torch.mps.driver_allocated_memory() / 1e9
        return stats

    def __repr__(self) -> str:
        caps = []
        if self.supports_amp:
            caps.append("amp")
        if self.supports_flash_attention:
            caps.append("flash_attn")
        if self.supports_svd:
            caps.append("svd")
        return f"DeviceManager(device={self.device}, capabilities=[{', '.join(caps)}])"


def get_best_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
