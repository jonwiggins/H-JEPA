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

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from ..utils.scheduler import create_lr_scheduler, create_ema_scheduler
from ..utils.checkpoint import CheckpointManager
from ..utils.logging import MetricsLogger, ProgressTracker

logger = logging.getLogger(__name__)


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
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        masking_fn: Any,
        config: Dict[str, Any],
        device: str = "cuda",
        resume_checkpoint: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.masking_fn = masking_fn
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Training config
        self.epochs = config['training']['epochs']
        self.warmup_epochs = config['training'].get('warmup_epochs', 0)
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        self.clip_grad = config['training'].get('clip_grad', None)
        self.use_amp = config['training'].get('use_amp', False)

        # Steps per epoch
        self.steps_per_epoch = len(train_loader) // self.accumulation_steps

        # Learning rate scheduler
        self.lr_scheduler = create_lr_scheduler(
            optimizer_type=config['training']['optimizer'],
            base_lr=config['training']['lr'],
            min_lr=config['training']['min_lr'],
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=self.warmup_epochs,
            schedule_type=config['training'].get('lr_schedule', 'cosine'),
        )

        # EMA scheduler for target encoder
        self.ema_scheduler = create_ema_scheduler(
            base_momentum=config['model']['ema']['momentum'],
            final_momentum=config['model']['ema']['momentum_end'],
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=config['model']['ema'].get('momentum_warmup_epochs', 0),
        )

        # Mixed precision training
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config['checkpoint']['checkpoint_dir'],
            keep_best_n=config['checkpoint'].get('keep_best_n', 3),
            save_frequency=config['checkpoint'].get('save_frequency', 10),
            metric_name='val_loss',
            mode='min',
        )

        # Metrics logger
        wandb_config = config['logging'].get('wandb', {})
        tensorboard_config = config['logging'].get('tensorboard', {})

        self.metrics_logger = MetricsLogger(
            experiment_name=config['logging']['experiment_name'],
            log_dir=config['logging']['log_dir'],
            config=config,
            use_wandb=wandb_config.get('enabled', False),
            use_tensorboard=tensorboard_config.get('enabled', True),
            wandb_project=wandb_config.get('project', 'h-jepa'),
            wandb_entity=wandb_config.get('entity', None),
            wandb_tags=wandb_config.get('tags', []),
        )

        # Progress tracker
        self.progress_tracker = ProgressTracker(
            total_epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Logging frequency
        self.log_frequency = config['logging'].get('log_frequency', 100)

        # Resume from checkpoint if provided
        if resume_checkpoint:
            self._resume_from_checkpoint(resume_checkpoint)

        logger.info("HJEPATrainer initialized")
        logger.info(f"Training for {self.epochs} epochs")
        logger.info(f"Steps per epoch: {self.steps_per_epoch}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")

    def train(self):
        """
        Main training loop.

        Runs training for the configured number of epochs,
        with validation and checkpointing.
        """
        logger.info("Starting training...")

        for epoch in range(self.current_epoch, self.epochs):
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
                val_loss = val_metrics['loss']
                is_best = self.checkpoint_manager.update_best_metric(val_loss)
            else:
                val_loss = None
                is_best = False

            # Save checkpoint
            if self.checkpoint_manager.should_save(epoch) or is_best:
                self._save_checkpoint(epoch, val_loss, is_best)

            # Log system metrics
            if epoch % 10 == 0:
                self.metrics_logger.log_system_metrics(step=self.global_step)

            # Print epoch summary
            val_metrics_to_print = (
                val_metrics if self.val_loader else None
            )
            self._print_epoch_summary(
                epoch, train_metrics, val_metrics_to_print
            )

        logger.info("Training completed!")
        self.metrics_logger.finish()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
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

        for batch_idx, batch in enumerate(pbar):
            # Compute effective step (accounting for accumulation)
            effective_step = batch_idx // self.accumulation_steps

            # Forward pass and loss computation
            loss, loss_dict = self._train_step(batch, epoch, effective_step)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (if accumulation is complete)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.clip_grad is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad,
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update learning rate
                lr = self.lr_scheduler(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Update global step
                self.global_step += 1

            # Accumulate metrics
            self.metrics_logger.accumulate_metrics(loss_dict)

            # Log metrics
            if batch_idx % self.log_frequency == 0:
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}",
                })

                # Log to W&B/TensorBoard
                if batch_idx % (self.log_frequency * 5) == 0:
                    log_dict = {
                        'lr': current_lr,
                        'ema_momentum': self.ema_scheduler(self.global_step),
                    }
                    log_dict.update(loss_dict)
                    self.metrics_logger.log_metrics(
                        log_dict,
                        step=self.global_step,
                        prefix="train/",
                    )

        # Compute epoch averages
        self.metrics_logger.log_accumulated_metrics(
            step=self.global_step,
            prefix="train_epoch/",
            reset=True,
        )

        # Return average metrics
        avg_loss = np.mean([m['loss'] for m in [loss_dict]])
        return {'loss': avg_loss}

    def _train_step(
        self,
        batch: Any,
        epoch: int,
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        masks_dict = self.masking_fn(
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
        target_masks = masks_dict['level_0']['targets']

        # Combine all target masks using OR operation (any target that covers a patch)
        # This gives us a single mask of shape [B, N] where 1 = predict, 0 = don't predict
        prediction_mask = target_masks.any(dim=1)  # [B, N] - positions to predict

        # Forward pass with automatic mixed precision
        with autocast(enabled=self.use_amp):
            # Forward through H-JEPA model
            outputs = self.model(images, prediction_mask)

            # Extract predictions and targets for all hierarchy levels
            predictions = outputs['predictions']
            targets = outputs['targets']

            # Compute loss
            # Loss function returns a dict with 'loss' key and other metrics
            loss_dict = self.loss_fn(
                predictions=predictions,
                targets=targets,
            )
            loss = loss_dict['loss']

        # Update target encoder with EMA
        ema_momentum = self.ema_scheduler(self.global_step)
        self._update_target_encoder(ema_momentum)

        # Monitor representation collapse
        if step % (self.log_frequency * 10) == 0:
            collapse_metrics = self._compute_collapse_metrics(
                outputs['context_features'],
                outputs['target_features'],
            )
            loss_dict.update(collapse_metrics)

        return loss, loss_dict

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
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
            context_masks, target_masks = self.masking_fn(
                batch_size=images.size(0),
                device=self.device,
            )

            # Forward pass
            with autocast(enabled=self.use_amp):
                context_embeddings = self.model.encode_context(images, context_masks)
                target_embeddings = self.model.encode_target(images, target_masks)

                predictions = self.model.predict(
                    context_embeddings,
                    target_masks,
                    context_masks,
                )

                loss, _ = self.loss_fn(
                    predictions=predictions,
                    targets=target_embeddings,
                )

            val_losses.append(loss.item())
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        avg_val_loss = np.mean(val_losses)
        return {'loss': avg_val_loss}

    @torch.no_grad()
    def _update_target_encoder(self, momentum: float):
        """
        Update target encoder parameters using EMA.

        target_params = momentum * target_params + (1 - momentum) * context_params

        Args:
            momentum: EMA momentum coefficient
        """
        if not hasattr(self.model, 'target_encoder') or not hasattr(self.model, 'context_encoder'):
            # If model doesn't have separate encoders, skip EMA update
            return

        context_params = self.model.context_encoder.parameters()
        target_params = self.model.target_encoder.parameters()

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
    ) -> Dict[str, float]:
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
        metrics['context_std'] = context_flat.std().item()
        metrics['target_std'] = target_flat.std().item()

        # Mean L2 norm
        metrics['context_norm'] = context_flat.norm(dim=1).mean().item()
        metrics['target_norm'] = target_flat.norm(dim=1).mean().item()

        # Effective rank (using SVD - expensive, so subsample if needed)
        if context_flat.size(0) > 100:
            context_sample = context_flat[:100]
            target_sample = target_flat[:100]
        else:
            context_sample = context_flat
            target_sample = target_flat

        try:
            # Compute singular values
            context_sv = torch.svd(context_sample)[1]
            target_sv = torch.svd(target_sample)[1]

            # Effective rank (normalized entropy of singular values)
            context_sv_norm = context_sv / context_sv.sum()
            target_sv_norm = target_sv / target_sv.sum()

            context_entropy = -(context_sv_norm * torch.log(context_sv_norm + 1e-8)).sum()
            target_entropy = -(target_sv_norm * torch.log(target_sv_norm + 1e-8)).sum()

            metrics['context_rank'] = torch.exp(context_entropy).item()
            metrics['target_rank'] = torch.exp(target_entropy).item()
        except:
            # SVD can fail, just skip rank computation
            pass

        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: Optional[float],
        is_best: bool,
    ):
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss (if available)
            is_best: Whether this is the best model so far
        """
        metrics = {'epoch': epoch}
        if val_loss is not None:
            metrics['val_loss'] = val_loss

        # Prepare scheduler state (as dict of values)
        scheduler_state = {
            'lr': self.lr_scheduler(self.global_step),
            'ema_momentum': self.ema_scheduler(self.global_step),
        }

        self.checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=scheduler_state,
            scaler=self.scaler,
            metrics=metrics,
            extra_state={'global_step': self.global_step},
            is_best=is_best,
        )

    def _resume_from_checkpoint(self, checkpoint_path: str):
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

        self.current_epoch = metadata['epoch'] + 1
        self.best_val_loss = metadata.get('best_metric', float('inf'))

        logger.info(f"Resumed from epoch {metadata['epoch']}")
        if self.best_val_loss != float('inf'):
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ):
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
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Optimizer instance
    """
    optimizer_type = config['training']['optimizer'].lower()
    lr = config['training']['lr']
    weight_decay = config['training'].get('weight_decay', 0.0)

    if optimizer_type == 'adamw':
        betas = config['training'].get('betas', [0.9, 0.95])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
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
