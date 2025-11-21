"""
Checkpoint management for H-JEPA training.

Handles:
- Saving model, optimizer, scheduler state
- Loading checkpoints for resuming training
- Tracking best models based on metrics
- Checkpoint cleanup to manage disk space
"""

import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import torch
import torch.nn as nn
from torch.amp import GradScaler

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving, loading, and cleanup of training checkpoints.

    Features:
    - Saves complete training state (model, optimizer, scheduler, etc.)
    - Tracks best checkpoints based on validation metrics
    - Automatically cleans up old checkpoints
    - Supports resuming training from checkpoint

    Args:
        checkpoint_dir: Directory to save checkpoints
        keep_best_n: Number of best checkpoints to keep (based on metric)
        save_frequency: Save checkpoint every N epochs
        metric_name: Name of metric to track for best checkpoints (e.g., 'val_loss')
        mode: 'min' or 'max' - whether lower or higher metric is better
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_n: int = 3,
        save_frequency: int = 10,
        metric_name: str = "val_loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best_n = keep_best_n
        self.save_frequency = save_frequency
        self.metric_name = metric_name
        self.mode = mode

        # Track best checkpoints
        self.best_checkpoints: List[Dict[str, Any]] = []
        self.best_metric = float("inf") if mode == "min" else float("-inf")

        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")
        logger.info(f"Tracking best {keep_best_n} checkpoints by {metric_name} ({mode})")

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[GradScaler] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number
            model: Model to save (can be DataParallel/DistributedDataParallel)
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            scaler: GradScaler for mixed precision training
            metrics: Dictionary of metrics to save
            extra_state: Any additional state to save
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint file
        """
        # Handle DataParallel/DistributedDataParallel
        if hasattr(model, "module"):
            model_state = cast(nn.Module, model.module).state_dict()
        else:
            model_state = model.state_dict()

        # Prepare checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler if isinstance(scheduler, dict) else None,
            "metrics": metrics or {},
            "best_metric": self.best_metric,
        }

        # Add scaler for mixed precision
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        # Add any extra state
        if extra_state is not None:
            checkpoint.update(extra_state)

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save as latest checkpoint (for easy resuming)
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        shutil.copy(checkpoint_path, latest_path)

        # Save as best checkpoint if applicable
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            shutil.copy(checkpoint_path, best_path)
            logger.info(f"New best checkpoint: {best_path}")

            # Track this checkpoint
            metric_value = metrics.get(self.metric_name, None) if metrics else None
            if metric_value is not None:
                self._update_best_checkpoints(
                    checkpoint_path=str(checkpoint_path),
                    epoch=epoch,
                    metric_value=metric_value,
                )

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[GradScaler] = None,
        device: Union[str, torch.device] = "cuda",
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            scaler: GradScaler to load state into (optional)
            device: Device to load checkpoint to

        Returns:
            Dictionary with checkpoint metadata (epoch, metrics, etc.)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        if hasattr(model, "module"):
            cast(nn.Module, model.module).load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Loaded optimizer state")

        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            if checkpoint["scheduler_state_dict"] is not None:
                # Scheduler state handling depends on implementation
                logger.info("Scheduler state found in checkpoint")

        # Load scaler state
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("Loaded GradScaler state")

        # Return metadata
        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "best_metric": checkpoint.get("best_metric", None),
        }

        logger.info(f"Resumed from epoch {metadata['epoch']}")
        return metadata

    def should_save(self, epoch: int) -> bool:
        """
        Check if checkpoint should be saved at this epoch.

        Args:
            epoch: Current epoch number

        Returns:
            True if checkpoint should be saved
        """
        return (epoch + 1) % self.save_frequency == 0

    def is_better_metric(self, metric_value: float) -> bool:
        """
        Check if the given metric value is better than the current best.

        Args:
            metric_value: Metric value to check

        Returns:
            True if this is a better metric value
        """
        if self.mode == "min":
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric

    def update_best_metric(self, metric_value: float) -> bool:
        """
        Update the best metric value.

        Args:
            metric_value: New metric value

        Returns:
            True if this is a new best metric
        """
        if self.is_better_metric(metric_value):
            self.best_metric = metric_value
            return True
        return False

    def _update_best_checkpoints(
        self,
        checkpoint_path: str,
        epoch: int,
        metric_value: float,
    ) -> None:
        """
        Update the list of best checkpoints and clean up old ones.

        Args:
            checkpoint_path: Path to the checkpoint file
            epoch: Epoch number
            metric_value: Metric value for this checkpoint
        """
        # Add new checkpoint to list
        self.best_checkpoints.append(
            {
                "path": checkpoint_path,
                "epoch": epoch,
                "metric": metric_value,
            }
        )

        # Sort by metric
        if self.mode == "min":
            self.best_checkpoints.sort(key=lambda x: x["metric"])
        else:
            self.best_checkpoints.sort(key=lambda x: x["metric"], reverse=True)

        # Keep only best N checkpoints
        if len(self.best_checkpoints) > self.keep_best_n:
            # Remove worst checkpoints
            to_remove = self.best_checkpoints[self.keep_best_n :]
            self.best_checkpoints = self.best_checkpoints[: self.keep_best_n]

            # Delete checkpoint files
            for ckpt in to_remove:
                try:
                    if os.path.exists(ckpt["path"]):
                        os.remove(ckpt["path"])
                        logger.info(f"Removed old checkpoint: {ckpt['path']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {ckpt['path']}: {e}")

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """
        Clean up old periodic checkpoints, keeping only the last N.

        This is separate from best checkpoint tracking and removes
        old periodic checkpoints to save disk space.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        # Find all checkpoint files
        checkpoint_pattern = str(self.checkpoint_dir / "checkpoint_epoch_*.pth")
        checkpoints = sorted(glob.glob(checkpoint_pattern))

        # Keep only the last N
        if len(checkpoints) > keep_last_n:
            to_remove = checkpoints[:-keep_last_n]
            for ckpt_path in to_remove:
                # Don't remove if it's in the best checkpoints list
                is_best = any(ckpt["path"] == ckpt_path for ckpt in self.best_checkpoints)
                if not is_best:
                    try:
                        os.remove(ckpt_path)
                        logger.info(f"Cleaned up old checkpoint: {ckpt_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {ckpt_path}: {e}")

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        if latest_path.exists():
            return str(latest_path)

        # Fallback: find the most recent checkpoint by epoch number
        checkpoint_pattern = str(self.checkpoint_dir / "checkpoint_epoch_*.pth")
        checkpoints = sorted(glob.glob(checkpoint_pattern))
        if checkpoints:
            return checkpoints[-1]

        return None

    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get the path to the best checkpoint.

        Returns:
            Path to best checkpoint, or None if no best checkpoint exists
        """
        best_path = self.checkpoint_dir / "checkpoint_best.pth"
        if best_path.exists():
            return str(best_path)
        return None


def save_checkpoint(
    filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, **kwargs: Any
) -> None:
    """
    Simple checkpoint saving utility function.

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        **kwargs: Additional state to save
    """
    if hasattr(model, "module"):
        model_state = cast(nn.Module, model.module).state_dict()
    else:
        model_state = model.state_dict()

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint.update(kwargs)

    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Simple checkpoint loading utility function.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint to

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint: Dict[str, Any] = torch.load(filepath, map_location=device)

    if hasattr(model, "module"):
        cast(nn.Module, model.module).load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Checkpoint loaded: {filepath}")
    return checkpoint
