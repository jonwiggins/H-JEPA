"""
Logging utilities for H-JEPA training.

Supports:
- Weights & Biases (W&B) integration
- TensorBoard logging
- Metrics tracking and aggregation
- Image and visualization logging
- System monitoring (GPU, memory)
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional imports
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore[assignment, misc]
    logging.warning("tensorboard not available. Install with: pip install tensorboard")

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Handles logging to both W&B and TensorBoard with metrics aggregation.

    Features:
    - Unified interface for W&B and TensorBoard
    - Automatic metric aggregation and averaging
    - Image and histogram logging
    - System metrics (GPU usage, memory)
    - Configuration logging

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for logs
        config: Configuration dictionary to log
        use_wandb: Enable W&B logging
        use_tensorboard: Enable TensorBoard logging
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team)
        wandb_tags: Tags for W&B run
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        wandb_project: str = "h-jepa",
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE

        # Initialize W&B
        if self.use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    config=config,
                    tags=wandb_tags or [],
                    dir=str(self.log_dir),
                )
                logger.info(f"W&B initialized: {wandb_project}/{experiment_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        # Initialize TensorBoard
        if self.use_tensorboard:
            try:
                tb_dir = self.log_dir / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))  # type: ignore[no-untyped-call]
                logger.info(f"TensorBoard initialized: {tb_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False

        # Metrics aggregation
        self.metrics_buffer: Dict[str, List[Union[float, int, torch.Tensor]]] = defaultdict(list)
        self.step = 0

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        prefix: str = "",
        commit: bool = True,
    ) -> None:
        """
        Log metrics to W&B and TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step number (uses internal counter if None)
            prefix: Prefix to add to metric names (e.g., "train/", "val/")
            commit: Whether to commit the metrics (W&B)
        """
        if step is not None:
            self.step = step

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Log to W&B
        if self.use_wandb:
            try:
                wandb.log(metrics, step=self.step, commit=commit)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")

        # Log to TensorBoard
        if self.use_tensorboard:
            try:
                for name, value in metrics.items():
                    self.tb_writer.add_scalar(name, value, self.step)  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")

        self.step += 1

    def log_image(
        self,
        name: str,
        image: Union[npt.NDArray[np.float32], torch.Tensor],
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        Log an image to W&B and TensorBoard.

        Args:
            name: Name/key for the image
            image: Image tensor (C, H, W) or numpy array
            step: Global step number
            caption: Optional caption for the image
        """
        if step is None:
            step = self.step

        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # Log to W&B
        if self.use_wandb:
            try:
                wandb.log({name: wandb.Image(image, caption=caption)}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log image to W&B: {e}")

        # Log to TensorBoard
        if self.use_tensorboard:
            try:
                self.tb_writer.add_image(name, image, step)  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Failed to log image to TensorBoard: {e}")

    def log_images(
        self,
        name: str,
        images: List[Union[npt.NDArray[np.float32], torch.Tensor]],
        step: Optional[int] = None,
        captions: Optional[List[str]] = None,
    ) -> None:
        """
        Log multiple images to W&B and TensorBoard.

        Args:
            name: Name/key for the images
            images: List of image tensors
            step: Global step number
            captions: Optional list of captions
        """
        if step is None:
            step = self.step

        # Convert tensors to numpy
        images_np = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            images_np.append(img)

        # Log to W&B
        if self.use_wandb:
            try:
                wandb_images = [
                    wandb.Image(img, caption=captions[i] if captions else None)
                    for i, img in enumerate(images_np)
                ]
                wandb.log({name: wandb_images}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log images to W&B: {e}")

        # Log to TensorBoard (as grid)
        if self.use_tensorboard:
            try:
                # Stack images into a grid
                images_tensor = torch.stack(
                    [
                        torch.from_numpy(img) if isinstance(img, np.ndarray) else img
                        for img in images_np
                    ]
                )
                from torchvision.utils import make_grid

                grid = make_grid(images_tensor, nrow=4)
                self.tb_writer.add_image(name, grid, step)  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Failed to log images to TensorBoard: {e}")

    def log_histogram(
        self,
        name: str,
        values: Union[npt.NDArray[np.float32], torch.Tensor],
        step: Optional[int] = None,
    ) -> None:
        """
        Log a histogram of values.

        Args:
            name: Name for the histogram
            values: Values to create histogram from
            step: Global step number
        """
        if step is None:
            step = self.step

        # Convert to numpy if tensor
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        # Log to W&B
        if self.use_wandb:
            try:
                wandb.log(
                    {
                        name: wandb.Histogram(
                            values.tolist() if isinstance(values, np.ndarray) else values
                        )
                    },
                    step=step,
                )
            except Exception as e:
                logger.warning(f"Failed to log histogram to W&B: {e}")

        # Log to TensorBoard
        if self.use_tensorboard:
            try:
                self.tb_writer.add_histogram(name, values, step)  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.warning(f"Failed to log histogram to TensorBoard: {e}")

    def log_model_gradients(
        self,
        model: nn.Module,
        step: Optional[int] = None,
    ) -> None:
        """
        Log gradient histograms for model parameters.

        Args:
            model: PyTorch model
            step: Global step number
        """
        if step is None:
            step = self.step

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(
                    f"gradients/{name}",
                    param.grad,
                    step=step,
                )

    def log_model_weights(
        self,
        model: nn.Module,
        step: Optional[int] = None,
    ) -> None:
        """
        Log weight histograms for model parameters.

        Args:
            model: PyTorch model
            step: Global step number
        """
        if step is None:
            step = self.step

        for name, param in model.named_parameters():
            self.log_histogram(
                f"weights/{name}",
                param.data,
                step=step,
            )

    def log_hierarchical_losses(
        self,
        loss_dict: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "train/hierarchy/",
    ) -> None:
        """
        Log hierarchical losses per level with percentages.

        Args:
            loss_dict: Dictionary with keys like 'loss_level_0', 'loss_level_1', etc.
            step: Global step number
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step

        # Extract level losses
        level_losses: Dict[str, float] = {}
        total_loss: float = 0.0
        for key, value in loss_dict.items():
            if key.startswith("loss_level_"):
                level = key.split("_")[-1]
                level_losses[level] = value
                total_loss += value

        # Log individual level losses and percentages
        for level, loss_value in level_losses.items():
            self.log_metrics(
                {
                    f"{prefix}level_{level}_loss": loss_value,
                    f"{prefix}level_{level}_percentage": (
                        (loss_value / total_loss * 100) if total_loss > 0 else 0
                    ),
                },
                step=step,
                commit=False,
            )

    def log_prediction_comparison(
        self,
        images: torch.Tensor,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        masks: torch.Tensor,
        step: Optional[int] = None,
        max_images: int = 4,
    ) -> None:
        """
        Log side-by-side comparison of predictions vs targets.

        Args:
            images: Original images [B, C, H, W]
            predictions: List of predictions per hierarchy level
            targets: List of targets per hierarchy level
            masks: Prediction mask [B, N]
            step: Global step number
            max_images: Maximum number of images to log
        """
        if step is None:
            step = self.step

        batch_size = min(images.size(0), max_images)

        # Log original images
        self.log_images(
            "predictions/original_images",
            [images[i] for i in range(batch_size)],
            step=step,
        )

        # Log masked images (visualization of what's masked)
        try:
            # This is a placeholder - actual mask visualization would need patch info
            # For now, just log that masks are being used
            mask_ratio = masks.float().mean().item()
            self.log_metrics(
                {"predictions/mask_ratio": mask_ratio},
                step=step,
                commit=False,
            )
        except Exception as e:
            logger.warning(f"Failed to log mask visualization: {e}")

        # Log predictions vs targets for each hierarchy level
        for level_idx, (preds, targs) in enumerate(zip(predictions, targets)):
            # Compute similarity
            similarity = (
                F.cosine_similarity(preds.flatten(1), targs.flatten(1), dim=1).mean().item()
            )

            self.log_metrics(
                {f"predictions/level_{level_idx}_similarity": similarity},
                step=step,
                commit=False,
            )

    def log_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        tag: str = "embeddings",
        step: Optional[int] = None,
    ) -> None:
        """
        Log embeddings for t-SNE/UMAP visualization in TensorBoard.

        Args:
            embeddings: Embedding vectors [N, D]
            labels: Optional labels for each embedding [N]
            tag: Tag for the embedding visualization
            step: Global step number
        """
        if step is None:
            step = self.step

        # TensorBoard embedding visualization
        if self.use_tensorboard:
            try:
                # Convert to numpy
                emb_np = embeddings.detach().cpu()

                # Prepare metadata (labels)
                metadata = None
                if labels is not None:
                    metadata = labels.detach().cpu().numpy().tolist()

                self.tb_writer.add_embedding(  # type: ignore[no-untyped-call]
                    emb_np,
                    metadata=metadata,
                    tag=tag,
                    global_step=step,
                )
                logger.info(f"Logged embeddings: {tag} at step {step}")
            except Exception as e:
                logger.warning(f"Failed to log embeddings: {e}")

    def accumulate_metrics(
        self,
        metrics: Dict[str, float],
    ) -> None:
        """
        Accumulate metrics for later averaging.

        Useful for accumulating metrics over an epoch before logging.

        Args:
            metrics: Dictionary of metric names and values
        """
        for name, value in metrics.items():
            self.metrics_buffer[name].append(value)

    def log_accumulated_metrics(
        self,
        step: Optional[int] = None,
        prefix: str = "",
        reset: bool = True,
    ) -> None:
        """
        Log averaged accumulated metrics and optionally reset buffer.

        Args:
            step: Global step number
            prefix: Prefix for metric names
            reset: Whether to reset the accumulation buffer
        """
        if not self.metrics_buffer:
            return

        # Compute averages
        averaged_metrics: Dict[str, Union[float, int]] = {}
        for name, values in self.metrics_buffer.items():
            # Convert to CPU if tensors, then to numpy array
            import torch

            cpu_values = [v.cpu().item() if isinstance(v, torch.Tensor) else v for v in values]
            averaged_metrics[name] = float(np.mean(cpu_values))

        # Log the averaged metrics
        self.log_metrics(averaged_metrics, step=step, prefix=prefix)

        # Reset buffer if requested
        if reset:
            self.metrics_buffer.clear()

    def log_system_metrics(
        self,
        step: Optional[int] = None,
    ) -> None:
        """
        Log system metrics (GPU usage, memory, etc.).

        Args:
            step: Global step number
        """
        if step is None:
            step = self.step

        metrics = {}

        # GPU metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # Memory
                metrics[f"system/gpu{i}_memory_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
                metrics[f"system/gpu{i}_memory_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9

                # Utilization (if available)
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f"system/gpu{i}_utilization"] = util.gpu
                    metrics[f"system/gpu{i}_memory_utilization"] = util.memory
                except Exception:
                    # GPU monitoring may fail for various reasons (drivers, permissions, etc)
                    # Continue silently as this is optional telemetry
                    pass

        self.log_metrics(metrics, step=step, prefix="")

    def watch_model(
        self,
        model: nn.Module,
        log_freq: int = 1000,
    ) -> None:
        """
        Watch model for gradient and parameter tracking (W&B).

        Args:
            model: Model to watch
            log_freq: Logging frequency
        """
        if self.use_wandb:
            try:
                wandb.watch(model, log="all", log_freq=log_freq)
                logger.info("Model watching enabled in W&B")
            except Exception as e:
                logger.warning(f"Failed to watch model in W&B: {e}")

    def finish(self) -> None:
        """Clean up logging resources."""
        if self.use_wandb:
            try:
                wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"Error finishing W&B: {e}")

        if self.use_tensorboard:
            try:
                self.tb_writer.close()  # type: ignore[no-untyped-call]
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"Error closing TensorBoard: {e}")

    def __enter__(self) -> "MetricsLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.finish()


class ProgressTracker:
    """
    Tracks training progress with timing and ETA estimation.

    Args:
        total_epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch
    """

    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
    ):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch

        self.start_time = time.time()
        self.epoch_start_time: Optional[float] = None
        self.step_times: List[float] = []

    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()

    def step(self) -> None:
        """Record a training step."""
        if self.epoch_start_time is not None:
            step_time = time.time() - self.epoch_start_time
            self.step_times.append(step_time)

            # Keep only recent steps for ETA calculation
            if len(self.step_times) > 100:
                self.step_times.pop(0)

    def get_eta(self, current_epoch: int, current_step: int) -> str:
        """
        Get estimated time remaining.

        Args:
            current_epoch: Current epoch number
            current_step: Current step within epoch

        Returns:
            Formatted ETA string
        """
        if not self.step_times:
            return "N/A"

        avg_step_time = float(np.mean(self.step_times))
        steps_remaining = (self.total_epochs - current_epoch - 1) * self.steps_per_epoch + (
            self.steps_per_epoch - current_step
        )
        eta_seconds = avg_step_time * steps_remaining

        return self._format_time(eta_seconds)

    def get_elapsed_time(self) -> str:
        """
        Get total elapsed time.

        Returns:
            Formatted elapsed time string
        """
        elapsed = time.time() - self.start_time
        return self._format_time(elapsed)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup Python logging configuration.

    Args:
        log_file: Optional file path to save logs
        level: Logging level
    """
    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
