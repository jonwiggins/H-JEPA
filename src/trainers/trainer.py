"""
H-JEPA Trainer - Complete training loop implementation.

Features:
- Full training loop with forward pass, loss computation, backpropagation
- EMA updates for target encoder
- Gradient clipping and mixed precision training
- Validation loop
- Checkpoint saving/loading
- W&B and TensorBoard integration
- Collapse monitoring (variance, rank metrics)
- Distributed training support (future)
"""

import logging
import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.checkpoint import CheckpointManager
from ..utils.device import DeviceManager
from ..utils.logging import MetricsLogger, ProgressTracker
from ..utils.scheduler import create_ema_scheduler, create_lr_scheduler

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to halt training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss-like metrics (lower is better),
              'max' for accuracy-like metrics (higher is better)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric_value: Current value of the monitored metric

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False

        improved = (
            (metric_value < self.best_value - self.min_delta)
            if self.mode == "min"
            else (metric_value > self.best_value + self.min_delta)
        )

        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class HJEPATrainer:
    """
    Main trainer class for H-JEPA.

    Handles:
    - Training loop with masking and prediction
    - EMA updates for target encoder
    - Loss computation with hierarchical weights
    - Checkpointing and logging
    - Collapse prevention monitoring

    Args:
        model: H-JEPA model (includes context encoder, target encoder, predictor)
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer for training
        loss_fn: Loss function
        masking_fn: Masking function for creating context/target masks
        config: Training configuration dictionary
        device: Device to train on
        resume_checkpoint: Path to checkpoint to resume from (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        masking_fn: Callable[[int, str], dict[str, Any]],
        config: dict[str, Any],
        device: str | torch.device = "cuda",
        resume_checkpoint: str | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.masking_fn = masking_fn
        self.config = config
        # Device management
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device

        # Move model to device
        self.model = self.model.to(self.device)

        # Training config
        self.epochs = config["training"]["epochs"]
        self.warmup_epochs = config["training"].get("warmup_epochs", 0)
        self.accumulation_steps = config["training"].get("accumulation_steps", 1)
        self.clip_grad = config["training"].get("clip_grad", None)
        self.use_amp = config["training"].get("use_amp", False)

        # Steps per epoch
        if self.accumulation_steps < 1 or len(train_loader) == 0:
            raise ValueError("Training requires a nonempty loader and positive accumulation_steps")
        self.steps_per_epoch = math.ceil(len(train_loader) / self.accumulation_steps)
        self.max_steps = config["training"].get("max_steps")

        # Learning rate scheduler
        self.lr_scheduler: Callable[[int], float] = create_lr_scheduler(
            optimizer_type=config["training"]["optimizer"],
            base_lr=config["training"].get("lr", config["training"].get("learning_rate", 1e-4)),
            min_lr=config["training"].get("scheduler_params", {}).get("min_lr", 1e-6),
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=self.warmup_epochs,
            schedule_type=config["training"].get("lr_schedule", "cosine"),
        )

        # EMA scheduler for target encoder
        ema_config = config.get("training", {}).get("ema_momentum_schedule", {})
        model_ema = config.get("model", {}).get("ema", {})
        self.ema_scheduler = create_ema_scheduler(
            base_momentum=ema_config.get("start", model_ema.get("momentum", 0.996)),
            final_momentum=ema_config.get("end", model_ema.get("momentum_end", 1.0)),
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=model_ema.get("momentum_warmup_epochs", 0),
        )
        if "warmup_steps" in ema_config:
            self.ema_scheduler.warmup_steps = ema_config["warmup_steps"]

        # Mixed precision training (only supported on CUDA)
        if not self.device_manager.supports_amp and self.use_amp:
            logger.warning(
                f"Mixed precision training not supported on {self.device.type}, disabling AMP"
            )
            self.use_amp = False
        self.scaler: GradScaler | None = GradScaler(device="cuda") if self.use_amp else None

        # Checkpoint manager
        checkpoint_dir = config.get("checkpoint", {}).get(
            "checkpoint_dir", config.get("experiment", {}).get("output_dir", "checkpoints")
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_best_n=config.get("checkpoint", {}).get("keep_best_n", 3),
            save_frequency=config.get("checkpoint", {}).get("save_frequency", 10),
            metric_name=config.get("checkpoint", {}).get("metric", "val_loss"),
            mode=config.get("checkpoint", {}).get("mode", "min"),
        )

        # Metrics logger
        wandb_config = config["logging"].get("wandb", {})
        tensorboard_config = config["logging"].get("tensorboard", {})

        self.metrics_logger = MetricsLogger(
            experiment_name=config.get("logging", {}).get(
                "experiment_name", config.get("experiment", {}).get("name", "hjepa")
            ),
            log_dir=config.get("logging", {}).get(
                "log_dir", config.get("experiment", {}).get("output_dir", "logs")
            ),
            config=config,
            use_wandb=config.get("logging", {}).get(
                "use_wandb", wandb_config.get("enabled", False)
            ),
            use_tensorboard=config.get("logging", {}).get(
                "use_tensorboard", tensorboard_config.get("enabled", True)
            ),
            wandb_project=wandb_config.get("project", "h-jepa"),
            wandb_entity=wandb_config.get("entity", None),
            wandb_tags=wandb_config.get("tags", []),
        )

        # Progress tracker
        self.progress_tracker = ProgressTracker(
            total_epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Logging frequency
        self.log_frequency = config["logging"].get("log_frequency", 100)

        # Early stopping
        early_stop_config = config.get("training", {}).get("early_stopping", {})
        if early_stop_config.get("enabled", False):
            self.early_stopping: EarlyStopping | None = EarlyStopping(
                patience=early_stop_config.get("patience", 10),
                min_delta=early_stop_config.get("min_delta", 0.0),
                mode=early_stop_config.get("mode", "min"),
            )
            logger.info(
                "Early stopping enabled: patience=%d, mode=%s",
                early_stop_config.get("patience", 10),
                early_stop_config.get("mode", "min"),
            )
        else:
            self.early_stopping = None

        # Online evaluation config
        online_eval_config = config.get("training", {}).get("online_eval", {})
        self.online_eval_enabled = online_eval_config.get("enabled", False)
        self.online_eval_frequency = online_eval_config.get("frequency", 5)
        self.online_eval_type = online_eval_config.get("type", "knn")
        self.online_eval_k = online_eval_config.get("k", 20)
        self.online_eval_max_samples = online_eval_config.get("max_samples", 5000)

        # Resume from checkpoint if provided
        if resume_checkpoint:
            self._resume_from_checkpoint(resume_checkpoint)

        logger.info("HJEPATrainer initialized")
        logger.info(f"Training for {self.epochs} epochs")
        logger.info(f"Steps per epoch: {self.steps_per_epoch}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")

    def train(self) -> None:
        """
        Main training loop.

        Runs training for the configured number of epochs,
        with validation and checkpointing.
        """
        logger.info("Starting training...")

        for epoch in range(self.current_epoch, self.epochs):
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break
            self.current_epoch = epoch
            self.progress_tracker.start_epoch()

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Log epoch metrics
            self.metrics_logger.log_metrics(
                train_metrics,
                step=self.global_step,
                prefix="train/",
            )

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch)
                self.metrics_logger.log_metrics(
                    val_metrics,
                    step=self.global_step,
                    prefix="val/",
                )

                # Check if best model
                val_loss = val_metrics["loss"]
                is_best = self.checkpoint_manager.update_best_metric(val_loss)
            else:
                val_loss = None
                is_best = False

            # Log system metrics
            if epoch % 10 == 0:
                self.metrics_logger.log_system_metrics(step=self.global_step)

            # Log prediction visualizations and embeddings periodically
            if epoch % 5 == 0:
                self._log_epoch_visualizations(epoch)

            # Online evaluation (k-NN or linear probe on frozen features)
            if (
                self.online_eval_enabled
                and self.val_loader is not None
                and (epoch + 1) % self.online_eval_frequency == 0
            ):
                online_metrics = self._run_online_eval()
                self.metrics_logger.log_metrics(
                    online_metrics,
                    step=self.global_step,
                    prefix="online_eval/",
                )

            stop_early = False
            if self.early_stopping is not None:
                metric = val_loss if val_loss is not None else train_metrics["loss"]
                stop_early = self.early_stopping(metric)
            reached_limit = self.max_steps is not None and self.global_step >= self.max_steps
            # Save after all stochastic epoch work and early-stopping state updates.
            if (
                self.checkpoint_manager.should_save(epoch)
                or is_best
                or stop_early
                or reached_limit
                or epoch + 1 == self.epochs
            ):
                self._save_checkpoint(epoch, val_loss, is_best)
            if stop_early:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

            # Print epoch summary
            val_metrics_to_print = val_metrics if self.val_loader else None
            self._print_epoch_summary(epoch, train_metrics, val_metrics_to_print)

        logger.info("Training completed!")
        self.metrics_logger.finish()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.metrics_logger.accumulate_metrics({})  # Reset accumulation

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            leave=True,
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        num_batches = 0
        window_samples = 0
        self.epoch_complete = False
        for batch_idx, batch in enumerate(pbar):
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break
            # Compute effective step (accounting for accumulation)
            effective_step = batch_idx // self.accumulation_steps

            # Forward pass and loss computation
            loss, loss_dict = self._train_step(batch, epoch, effective_step)

            total_loss += float(loss.detach())
            num_batches += 1
            # Accumulate sample-weighted means, then divide by the actual
            # window population. The last DataLoader batch can be smaller.
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            batch_samples = images.shape[0]
            window_samples += batch_samples
            loss = loss * batch_samples

            # Backward pass
            if self.use_amp and self.scaler is not None:
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()  # type: ignore[no-untyped-call]
            else:
                loss.backward()  # type: ignore[no-untyped-call]

            # Optimizer step (if accumulation is complete)
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx + 1 == len(
                self.train_loader
            ):
                # The schedule value belongs to the update about to be applied.
                lr = self.lr_scheduler(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.div_(window_samples)
                window_samples = 0
                # Gradient clipping
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad,
                    )

                # A skipped AMP update must not advance the EMA or schedules.
                updated = True
                if self.use_amp and self.scaler is not None:
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    updated = self.scaler.get_scale() >= old_scale
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if updated:
                    self._update_target_encoder(self.ema_scheduler(self.global_step))
                    self.global_step += 1

            # Accumulate metrics (before memory cleanup to avoid UnboundLocalError)
            self.metrics_logger.accumulate_metrics(loss_dict)

            # Clear memory cache periodically to prevent leaks (especially on MPS)
            if self.global_step % 50 == 0:
                self.device_manager.empty_cache()

            # Log metrics
            if batch_idx % self.log_frequency == 0:
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss_dict['loss']:.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

                # Log to W&B/TensorBoard
                if batch_idx % (self.log_frequency * 5) == 0:
                    log_dict = {
                        "lr": current_lr,
                        "ema_momentum": self.ema_scheduler(self.global_step),
                    }
                    log_dict.update(loss_dict)

                    # Add memory logging for debugging
                    mem_stats = self.device_manager.memory_stats()
                    for key, val in mem_stats.items():
                        log_dict[f"memory_{self.device.type}_{key}"] = val
                    self.metrics_logger.log_metrics(
                        log_dict,
                        step=self.global_step,
                        prefix="train/",
                    )

                    # Log hierarchical losses if available
                    self.metrics_logger.log_hierarchical_losses(
                        loss_dict,
                        step=self.global_step,
                    )

                # Log gradient histograms periodically
                if batch_idx % (self.log_frequency * 20) == 0:
                    self.metrics_logger.log_model_gradients(
                        self.model,
                        step=self.global_step,
                    )

        self.epoch_complete = num_batches == len(self.train_loader)

        # Compute epoch averages
        self.metrics_logger.log_accumulated_metrics(
            step=self.global_step,
            prefix="train_epoch/",
            reset=True,
        )

        # Return average metrics
        # loss_dict["loss"] is always a float, not a tensor
        return {"loss": total_loss / max(1, num_batches)}

    def _train_step(
        self,
        batch: Any,
        epoch: int,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Single training step with forward pass and loss computation.

        Args:
            batch: Input batch (images)
            epoch: Current epoch
            step: Current step

        Returns:
            Tuple of (loss tensor, metrics dictionary)
        """
        # Get images from batch
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        images = images.to(self.device)

        # Generate masks
        masks_dict = self.masking_fn(  # type: ignore[call-arg]
            batch_size=images.size(0),
            device=self.device,
        )

        # Use level 0 masks (finest level)
        # The masking function returns:
        # - 'context': [B, N] - regions to keep visible
        # - 'targets': [B, num_target_masks, N] - multiple target regions to predict
        #
        # For H-JEPA, we need to combine all target masks into a single mask of patches to predict
        # targets shape: [B, num_target_masks, N]
        target_masks = masks_dict["level_0"]["targets"]

        # Combine all target masks using OR operation (any target that covers a patch)
        # This gives us a single mask of shape [B, N] where 1 = predict, 0 = don't predict
        prediction_mask = target_masks.any(dim=1)  # [B, N] - positions to predict

        # Forward pass with automatic mixed precision
        # Use appropriate device type for autocast
        device_type = self.device_manager.autocast_device_type
        with autocast(device_type=device_type, enabled=self.use_amp):
            # Forward through H-JEPA model
            outputs = self.model(
                images, prediction_mask, context_mask=masks_dict["level_0"].get("context")
            )

            # Extract predictions and targets for all hierarchy levels
            predictions = outputs["predictions"]
            targets = outputs["targets"]

            # Get context_features for VICReg collapse prevention
            # context_features are the encoder outputs (with gradients) that we want
            # to regularize to prevent representation collapse
            context_features = outputs.get("context_features", None)

            # Compute loss
            # Loss function returns a dict with 'loss' key and other metrics
            # Pass context_features so VICReg can be applied to encoder outputs
            loss_dict = self.loss_fn(
                predictions=predictions,
                targets=targets,
                context_features=context_features,
                masks=outputs.get("masks_valid"),
            )
            loss = loss_dict["loss"]

        # Monitor representation collapse
        if step % (self.log_frequency * 10) == 0:
            collapse_metrics = self._compute_collapse_metrics(
                outputs["context_features"],
                outputs["target_features"],
            )
            loss_dict.update(collapse_metrics)

        # Logging must not retain the autograd graph across the epoch. Keep the
        # differentiable loss separate from the scalar metrics promised above.
        metrics = {
            name: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
            for name, value in loss_dict.items()
        }
        return loss, metrics

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict[str, float]:
        """
        Validation loop for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        val_losses = []
        pbar = tqdm(
            self.val_loader,
            desc=f"Validation {epoch+1}/{self.epochs}",
            leave=False,
        )

        for batch in pbar:
            # Get images
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # Generate masks
            masks_dict = self.masking_fn(  # type: ignore[call-arg]
                batch_size=images.size(0),
                device=self.device,
            )

            # Extract target masks from level 0 and combine them
            target_masks = masks_dict["level_0"]["targets"]  # [B, num_target_masks, N]
            prediction_mask = target_masks.any(dim=1)  # [B, N] - positions to predict

            # Forward pass (same as training)
            device_type = self.device_manager.autocast_device_type
            with autocast(device_type=device_type, enabled=self.use_amp):
                # Forward through H-JEPA model
                outputs = self.model(
                    images, prediction_mask, context_mask=masks_dict["level_0"].get("context")
                )

                # Extract predictions and targets
                predictions = outputs["predictions"]
                targets = outputs["targets"]
                context_features = outputs.get("context_features", None)

                loss_dict = self.loss_fn(
                    predictions=predictions,
                    targets=targets,
                    context_features=context_features,
                    masks=outputs.get("masks_valid"),
                )
                loss = loss_dict["loss"]

            val_losses.append(loss.item())
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_val_loss = float(np.mean(val_losses))
        return {"loss": avg_val_loss}

    @torch.no_grad()
    def _update_target_encoder(self, momentum: float) -> None:
        """
        Update target encoder parameters using EMA.

        target_params = momentum * target_params + (1 - momentum) * context_params

        Args:
            momentum: EMA momentum coefficient
        """
        if not hasattr(self.model, "target_encoder") or not hasattr(self.model, "context_encoder"):
            # If model doesn't have separate encoders, skip EMA update
            return

        # End-to-end mode (LeWM-style): no EMA update, gradients flow through both branches.
        if getattr(self.model, "use_target_encoder", True) is False:
            return

        context_params = self.model.context_encoder.parameters()  # type: ignore[union-attr]
        target_params = self.model.target_encoder.parameters()  # type: ignore[union-attr]

        for target_param, context_param in zip(target_params, context_params):
            target_param.data.mul_(momentum).add_(
                context_param.data,
                alpha=1.0 - momentum,
            )

    @torch.no_grad()
    def _compute_collapse_metrics(
        self,
        context_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute metrics to monitor representation collapse.

        Metrics:
        - std: Standard deviation of embeddings (should not approach 0)
        - rank: Effective rank of embedding matrix (should be high)
        - mean_norm: Mean L2 norm of embeddings

        Args:
            context_embeddings: Context encoder embeddings
            target_embeddings: Target encoder embeddings

        Returns:
            Dictionary of collapse metrics
        """
        metrics = {}

        # Flatten embeddings
        if context_embeddings.dim() > 2:
            context_flat = context_embeddings.reshape(context_embeddings.size(0), -1)
        else:
            context_flat = context_embeddings

        if target_embeddings.dim() > 2:
            target_flat = target_embeddings.reshape(target_embeddings.size(0), -1)
        else:
            target_flat = target_embeddings

        # Standard deviation (should be > 0)
        metrics["context_std"] = context_flat.std(dim=0, correction=0).mean().item()
        metrics["target_std"] = target_flat.std(dim=0, correction=0).mean().item()

        # Mean L2 norm
        metrics["context_norm"] = context_flat.norm(dim=1).mean().item()
        metrics["target_norm"] = target_flat.norm(dim=1).mean().item()

        # Effective rank (using SVD - expensive, so subsample if needed)
        if context_flat.size(0) > 100:
            context_sample = context_flat[:100]
            target_sample = target_flat[:100]
        else:
            context_sample = context_flat
            target_sample = target_flat

        try:
            if not self.device_manager.supports_svd:
                metrics["context_eff_rank"] = -1
                metrics["target_eff_rank"] = -1
            else:
                # Compute singular values
                context_sv = torch.linalg.svdvals(context_sample - context_sample.mean(dim=0))
                target_sv = torch.linalg.svdvals(target_sample - target_sample.mean(dim=0))

                # Effective rank (normalized entropy of singular values)
                context_sv_norm = context_sv / context_sv.sum().clamp_min(1e-12)
                target_sv_norm = target_sv / target_sv.sum().clamp_min(1e-12)

                context_entropy = -(context_sv_norm * torch.log(context_sv_norm + 1e-8)).sum()
                target_entropy = -(target_sv_norm * torch.log(target_sv_norm + 1e-8)).sum()

                metrics["context_eff_rank"] = (
                    torch.exp(context_entropy).item() if context_sv.sum() > 0 else 0.0
                )
                metrics["target_eff_rank"] = (
                    torch.exp(target_entropy).item() if target_sv.sum() > 0 else 0.0
                )
        except RuntimeError:
            # SVD can fail, just skip rank computation
            pass

        return metrics

    @torch.no_grad()
    def _log_epoch_visualizations(self, epoch: int) -> None:
        """
        Log prediction visualizations and embeddings at epoch milestones.

        Args:
            epoch: Current epoch number
        """
        try:
            # Get a batch for visualization
            self.model.eval()
            batch = next(iter(self.train_loader))
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images[:4].to(self.device)  # Only use first 4 images

            # Generate masks
            masks_dict = self.masking_fn(  # type: ignore[call-arg]
                batch_size=images.size(0),
                device=self.device,
            )
            target_masks = masks_dict["level_0"]["targets"]
            prediction_mask = target_masks.any(dim=1)

            # Forward pass
            outputs = self.model(
                images, prediction_mask, context_mask=masks_dict["level_0"].get("context")
            )

            # Log prediction comparison
            self.metrics_logger.log_prediction_comparison(
                images=images,
                predictions=outputs["predictions"],
                targets=outputs["targets"],
                masks=prediction_mask,
                step=self.global_step,
                max_images=4,
            )

            # Log embeddings (context and target features)
            context_features = outputs["context_features"]
            target_features = outputs["target_features"]

            # Flatten and subsample if needed
            if context_features.dim() > 2:
                context_flat = context_features.reshape(context_features.size(0), -1)
            else:
                context_flat = context_features

            if target_features.dim() > 2:
                target_flat = target_features.reshape(target_features.size(0), -1)
            else:
                target_flat = target_features

            # Log context embeddings
            self.metrics_logger.log_embeddings(
                embeddings=context_flat,
                tag="embeddings/context",
                step=self.global_step,
            )

            # Log target embeddings
            self.metrics_logger.log_embeddings(
                embeddings=target_flat,
                tag="embeddings/target",
                step=self.global_step,
            )

            logger.info(f"Logged visualizations for epoch {epoch}")
            self.model.train()

        except (RuntimeError, OSError) as e:
            logger.warning(f"Failed to log epoch visualizations: {e}")
            self.model.train()

    @torch.no_grad()
    def _run_online_eval(self) -> dict[str, float]:
        """
        Run lightweight online evaluation (k-NN) during training.

        Returns:
            Dictionary of evaluation metrics
        """
        from ..evaluation.feature_extraction import extract_features

        self.model.eval()
        metrics: dict[str, float] = {}

        try:
            train_features, train_labels = extract_features(
                model=self.model,
                dataloader=self.train_loader,
                hierarchy_level=0,
                device=str(self.device),
                pool=True,
                normalize=True,
                max_samples=self.online_eval_max_samples,
                desc="Online eval (train)",
            )

            # val_loader is guaranteed non-None by the caller's guard
            val_features, val_labels = extract_features(
                model=self.model,
                dataloader=self.val_loader,  # type: ignore[arg-type]
                hierarchy_level=0,
                device=str(self.device),
                pool=True,
                normalize=True,
                max_samples=self.online_eval_max_samples,
                desc="Online eval (val)",
            )

            # Simple k-NN classification
            from sklearn.neighbors import KNeighborsClassifier

            knn = KNeighborsClassifier(n_neighbors=self.online_eval_k, metric="cosine", n_jobs=-1)
            knn.fit(train_features, train_labels)
            accuracy = knn.score(val_features, val_labels) * 100.0

            metrics["knn_accuracy"] = accuracy
            logger.info("Online k-NN eval: %.2f%% (k=%d)", accuracy, self.online_eval_k)

        except Exception as e:
            logger.warning("Online evaluation failed: %s", e)

        self.model.train()
        return metrics

    def _rng_state(self) -> dict[str, Any]:
        np_state = np.random.get_state(legacy=True)
        if not isinstance(np_state, tuple):
            raise ValueError("Checkpoint RNG capture requires NumPy's legacy MT19937 generator")
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if self.device.type == "mps":
            state["mps"] = torch.mps.get_rng_state()
        for name in ("train", "val"):
            loader = getattr(self, f"{name}_loader")
            if loader is not None and loader.generator is not None:
                state[name] = loader.generator.get_state()
        return state

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float | None,
        is_best: bool,
    ) -> str:
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss (if available)
            is_best: Whether this is the best model so far

        Returns:
            Path to the saved checkpoint file
        """
        metrics: dict[str, float] = {"epoch": float(epoch)}
        if val_loss is not None:
            metrics["val_loss"] = val_loss

        # Prepare scheduler state (as dict of values)
        scheduler_state = {
            "lr": self.lr_scheduler(self.global_step),
            "ema_momentum": self.ema_scheduler(self.global_step),
        }

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=scheduler_state,
            scaler=self.scaler,
            metrics=metrics,
            extra_state={
                "global_step": self.global_step,
                "config": self.config,
                "loss_state_dict": self.loss_fn.state_dict(),
                "rng_state": self._rng_state(),
                "epoch_complete": getattr(self, "epoch_complete", True),
                "early_stopping": (
                    vars(self.early_stopping) if self.early_stopping is not None else None
                ),
            },
            is_best=is_best,
        )

        return checkpoint_path

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            device=self.device,
        )

        if not metadata.get("epoch_complete", True):
            raise ValueError(
                "Cannot resume an incomplete epoch: sampler position is not saved. "
                "Use an epoch-boundary checkpoint for reproducible continuation."
            )
        if self.early_stopping is not None and metadata.get("early_stopping") is not None:
            self.early_stopping.__dict__.update(metadata["early_stopping"])
        self.current_epoch = metadata["epoch"] + 1
        self.global_step = metadata.get("global_step", self.current_epoch * self.steps_per_epoch)
        if "loss_state_dict" in metadata:
            self.loss_fn.load_state_dict(metadata["loss_state_dict"])
        if "rng_state" in metadata:
            state = metadata["rng_state"]
            random.setstate(state["python"])
            np_state = state["numpy"]
            np.random.set_state(
                (np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:])
            )
            torch.set_rng_state(state["torch"].cpu())
            if "cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
            if "mps" in state and self.device.type == "mps":
                torch.mps.set_rng_state(state["mps"].cpu())
            for name in ("train", "val"):
                loader = getattr(self, f"{name}_loader")
                if loader is not None and loader.generator is not None and name in state:
                    loader.generator.set_state(state[name].cpu())

        # Try to get best_metric from metadata, fallback to val_loss from metrics
        best_metric = metadata.get("best_metric", float("inf"))
        if best_metric == float("inf") and "metrics" in metadata:
            # If best_metric is not set, try to get val_loss from metrics
            val_loss = metadata["metrics"].get("val_loss", float("inf"))
            self.best_val_loss = val_loss
        else:
            self.best_val_loss = best_metric

        logger.info(f"Resumed from epoch {metadata['epoch']}")
        if self.best_val_loss != float("inf"):
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        """
        Print epoch summary.

        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        elapsed = self.progress_tracker.get_elapsed_time()
        eta = self.progress_tracker.get_eta(epoch, 0)

        summary = f"\n{'='*80}\n"
        summary += f"Epoch {epoch+1}/{self.epochs} Summary\n"
        summary += f"{'='*80}\n"
        summary += f"Train Loss: {train_metrics['loss']:.4f}\n"

        if val_metrics:
            summary += f"Val Loss:   {val_metrics['loss']:.4f}\n"

        summary += f"Elapsed:    {elapsed}\n"
        summary += f"ETA:        {eta}\n"
        summary += f"{'='*80}\n"

        logger.info(summary)


def create_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD:
    """
    Create optimizer from config.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Optimizer instance
    """
    optimizer_type = config["training"]["optimizer"].lower()
    lr = config["training"].get("lr", config["training"].get("learning_rate", 1e-4))
    weight_decay = config["training"].get("weight_decay", 0.0)

    optimizer: torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD
    if optimizer_type == "adamw":
        betas = config["training"].get("betas", [0.9, 0.95])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        momentum = config["training"].get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    logger.info(f"Created optimizer: {optimizer_type}")
    return optimizer
