"""
Enhanced TensorBoard logging for H-JEPA.

Provides specialized logging functions for:
- Hierarchical loss monitoring
- EMA dynamics tracking
- Representational collapse detection
- Prediction-target similarity analysis
- Gradient flow visualization
- Training stability monitoring

This module integrates with the existing MetricsLogger in src/utils/logging.py
to provide domain-specific H-JEPA metrics.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HJEPATensorBoardLogger:
    """
    Enhanced TensorBoard logging for H-JEPA training.

    Provides methods to log:
    - Hierarchical loss metrics
    - EMA encoder dynamics
    - Collapse indicators
    - Prediction quality metrics
    - Gradient flow statistics
    - Training stability metrics
    """

    def __init__(self, metrics_logger, num_hierarchies: int = 3):
        """
        Initialize H-JEPA TensorBoard logger.

        Args:
            metrics_logger: MetricsLogger instance from src/utils/logging.py
            num_hierarchies: Number of hierarchy levels (2-4)
        """
        self.metrics_logger = metrics_logger
        self.num_hierarchies = num_hierarchies

        # Buffer for tracking statistics
        self.loss_history = defaultdict(list)
        self.gradient_norm_history = defaultdict(list)
        self.num_clipped_steps = 0
        self.num_total_steps = 0

    # ========================================================================
    # 1. HIERARCHICAL LOSS MONITORING
    # ========================================================================

    def log_hierarchical_losses(
        self,
        loss_dict: Dict[str, torch.Tensor],
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log hierarchical loss metrics.

        Args:
            loss_dict: Dictionary from HJEPALoss containing:
                - 'loss': Total weighted loss
                - 'loss_h{i}': Loss at each hierarchy level
                - 'loss_unweighted': Mean unweighted loss
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics for logging
        """
        metrics = {}

        # Extract and log individual hierarchy losses
        total_loss = loss_dict["loss"].item()
        metrics["loss/total"] = total_loss

        # Log per-hierarchy losses
        for level in range(self.num_hierarchies):
            level_loss = loss_dict[f"loss_h{level}"].item()
            metrics[f"loss/h{level}"] = level_loss

            # Store for contribution calculation
            self.loss_history[f"loss_h{level}"].append(level_loss)

        # Log unweighted loss
        unweighted_loss = loss_dict["loss_unweighted"].item()
        metrics["loss/unweighted"] = unweighted_loss

        # Calculate and log loss contributions (percentage of total)
        if total_loss > 0:
            for level in range(self.num_hierarchies):
                level_loss = loss_dict[f"loss_h{level}"].item()
                contribution = 100.0 * level_loss / total_loss
                metrics[f"loss/contribution_h{level}"] = contribution

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    def get_loss_convergence_status(self) -> Dict[str, float]:
        """
        Compute loss convergence metrics (per-hierarchy convergence rate).

        Returns:
            Dictionary with convergence metrics per hierarchy level
        """
        metrics = {}

        if len(self.loss_history["loss_h0"]) < 10:
            return metrics  # Need history before computing

        for level in range(self.num_hierarchies):
            losses = self.loss_history[f"loss_h{level}"][-10:]
            if len(losses) >= 2:
                # Simple convergence rate: recent vs early
                convergence_rate = (losses[0] - losses[-1]) / (losses[0] + 1e-8)
                metrics[f"convergence/rate_h{level}"] = convergence_rate

        return metrics

    # ========================================================================
    # 2. EMA DYNAMICS MONITORING
    # ========================================================================

    def log_ema_dynamics(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        current_momentum: float,
        target_momentum: float,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log exponential moving average (EMA) dynamics.

        Args:
            context_encoder: Context encoder module
            target_encoder: Target encoder module (updated via EMA)
            current_momentum: Current EMA momentum value
            target_momentum: Target EMA momentum value (end of schedule)
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        # Log momentum schedule
        metrics["ema/momentum_current"] = current_momentum
        metrics["ema/momentum_target"] = target_momentum

        # Compute parameter divergence between encoders
        param_divergence = 0
        num_params = 0

        for p_context, p_target in zip(context_encoder.parameters(), target_encoder.parameters()):
            if p_context.requires_grad:
                divergence = torch.norm(p_context.data - p_target.data)
                param_divergence += divergence.item()
                num_params += 1

        if num_params > 0:
            avg_divergence = param_divergence / num_params
            metrics["ema/avg_parameter_divergence"] = avg_divergence

        # Compute weight magnitude ratio (target vs context)
        context_norm = 0
        target_norm = 0

        for p_context, p_target in zip(context_encoder.parameters(), target_encoder.parameters()):
            context_norm += torch.norm(p_context.data).item()
            target_norm += torch.norm(p_target.data).item()

        if context_norm > 0:
            weight_ratio = target_norm / context_norm
            metrics["ema/weight_magnitude_ratio"] = weight_ratio

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    # ========================================================================
    # 3. MASKING STRATEGY MONITORING
    # ========================================================================

    def log_masking_statistics(
        self,
        mask: torch.Tensor,
        num_patches: int,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log masking strategy statistics.

        Args:
            mask: Mask tensor [B, N] where 1 indicates masked position
            num_patches: Total number of patches in image
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        # Compute mask statistics
        batch_size = mask.shape[0]
        num_masked_per_sample = mask.sum(dim=1).float().mean()
        num_unmasked_per_sample = num_patches - num_masked_per_sample

        mask_ratio = num_masked_per_sample / num_patches

        metrics["masking/mask_ratio"] = mask_ratio.item()
        metrics["masking/num_masked_patches"] = num_masked_per_sample.item()
        metrics["masking/num_unmasked_patches"] = num_unmasked_per_sample.item()

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    # ========================================================================
    # 4. PREDICTION QUALITY ANALYSIS
    # ========================================================================

    def log_prediction_quality(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log prediction quality metrics comparing predictions to targets.

        Args:
            predictions: List of prediction tensors [B, N_level, D], one per hierarchy
            targets: List of target tensors [B, N_level, D], one per hierarchy
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        for level in range(len(predictions)):
            pred = predictions[level]  # [B, N, D]
            targ = targets[level]  # [B, N, D]

            # Normalize for cosine similarity
            pred_norm = F.normalize(pred, dim=-1, eps=1e-8)
            targ_norm = F.normalize(targ, dim=-1, eps=1e-8)

            # Per-patch cosine similarity
            cosine_sim = (pred_norm * targ_norm).sum(dim=-1)  # [B, N]

            # Log statistics
            metrics[f"prediction/level{level}_cosine_sim_mean"] = cosine_sim.mean().item()
            metrics[f"prediction/level{level}_cosine_sim_std"] = cosine_sim.std().item()
            metrics[f"prediction/level{level}_cosine_sim_min"] = cosine_sim.min().item()
            metrics[f"prediction/level{level}_cosine_sim_max"] = cosine_sim.max().item()

            # MSE between normalized embeddings
            mse_loss = F.mse_loss(pred_norm, targ_norm)
            metrics[f"prediction/level{level}_normalized_mse"] = mse_loss.item()

            # L2 distance
            l2_dist = torch.norm(pred - targ, dim=-1).mean()
            metrics[f"prediction/level{level}_l2_distance"] = l2_dist.item()

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    # ========================================================================
    # 5. REPRESENTATIONAL COLLAPSE MONITORING
    # ========================================================================

    def log_collapse_metrics(
        self,
        features: torch.Tensor,
        level: int,
        global_step: int,
        prefix: str = "train/",
    ) -> Dict[str, float]:
        """
        Log metrics for detecting representational collapse.

        Collapse occurs when:
        - Features collapse to constant values (low variance)
        - Features have low rank (concentrated on few dimensions)
        - All samples produce similar embeddings (high similarity)

        Args:
            features: Feature tensor [B*N, D] or [N, D]
            level: Hierarchy level
            global_step: Current training step
            prefix: Logging prefix (train/val/)

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        # Normalize features
        features_norm = F.normalize(features, dim=-1, eps=1e-8)

        # 1. Standard deviation per dimension (collapse if std << mean)
        std_per_dim = features_norm.std(dim=0)
        mean_std = std_per_dim.mean()
        min_std = std_per_dim.min()

        metrics[f"collapse/level{level}_mean_std_per_dim"] = mean_std.item()
        metrics[f"collapse/level{level}_min_std_per_dim"] = min_std.item()

        # 2. Effective rank (via SVD)
        try:
            # For large matrices, use low-rank SVD approximation
            if features_norm.shape[0] > 10000:
                # Use randomized SVD
                U, S, V = torch.linalg.svd(features_norm[:10000], full_matrices=False)
            else:
                U, S, V = torch.linalg.svd(features_norm, full_matrices=False)

            S_norm = S / S.sum()
            entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
            effective_rank = torch.exp(entropy)

            metrics[f"collapse/level{level}_effective_rank"] = effective_rank.item()
        except Exception as e:
            logger.warning(f"Could not compute effective rank: {e}")

        # 3. Pairwise cosine similarity (collapse if high similarity)
        # Sample for efficiency
        sample_size = min(500, features_norm.shape[0])
        sample_indices = torch.randperm(features_norm.shape[0])[:sample_size]
        features_sample = features_norm[sample_indices]

        # Compute pairwise similarities
        similarity_matrix = features_sample @ features_sample.T
        # Get upper triangle (exclude diagonal which is always 1.0)
        triu_indices = torch.triu_indices(
            sample_size, sample_size, offset=1, device=features.device
        )
        similarities = similarity_matrix[triu_indices[0], triu_indices[1]]

        metrics[f"collapse/level{level}_mean_similarity"] = similarities.mean().item()
        metrics[f"collapse/level{level}_max_similarity"] = similarities.max().item()
        metrics[f"collapse/level{level}_std_similarity"] = similarities.std().item()

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix=prefix,
            commit=False,
        )

        return metrics

    # ========================================================================
    # 6. GRADIENT FLOW ANALYSIS
    # ========================================================================

    def log_gradient_flow(
        self,
        model: nn.Module,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log gradient flow statistics.

        Args:
            model: H-JEPA model (after backward pass)
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        # Collect gradient statistics
        total_grad_norm = 0
        num_params = 0

        gradient_stats = defaultdict(lambda: {"mean": 0, "max": 0, "min": float("inf")})

        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_abs = param.grad.abs()
                grad_mean = grad_abs.mean().item()
                grad_max = grad_abs.max().item()
                grad_min = grad_abs.min().item()

                metrics[f"gradients/{name}_mean"] = grad_mean
                metrics[f"gradients/{name}_max"] = grad_max

                # Accumulate for module-level stats
                module_name = name.split(".")[0]  # Get top-level module
                gradient_stats[module_name]["mean"] += grad_mean * param.numel()
                gradient_stats[module_name]["max"] = max(
                    gradient_stats[module_name]["max"], grad_max
                )
                gradient_stats[module_name]["min"] = min(
                    gradient_stats[module_name]["min"], grad_min
                )

                # Global gradient norm
                total_grad_norm += torch.norm(param.grad.data).item()
                num_params += 1

        # Module-level statistics
        for module_name, stats in gradient_stats.items():
            if stats["mean"] > 0:
                metrics[f"gradient_flow/{module_name}_mean"] = (
                    stats["mean"] / num_params if num_params > 0 else 0
                )
                metrics[f"gradient_flow/{module_name}_max"] = stats["max"]

        # Global gradient norm
        metrics["gradient_flow/global_norm"] = total_grad_norm

        # Store for history
        self.gradient_norm_history["global_norm"].append(total_grad_norm)

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    def log_gradient_clipping(
        self,
        global_norm_before: float,
        global_norm_after: float,
        clip_threshold: float,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log gradient clipping statistics.

        Args:
            global_norm_before: Global gradient norm before clipping
            global_norm_after: Global gradient norm after clipping
            clip_threshold: Clipping threshold
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        metrics["gradient/global_norm_before_clip"] = global_norm_before
        metrics["gradient/global_norm_after_clip"] = global_norm_after

        was_clipped = 1.0 if global_norm_before > clip_threshold else 0.0
        metrics["gradient/was_clipped"] = was_clipped

        # Clipping ratio
        clipping_ratio = global_norm_after / (global_norm_before + 1e-8)
        metrics["gradient/clipping_ratio"] = clipping_ratio

        # Track clipping statistics
        self.num_clipped_steps += was_clipped
        self.num_total_steps += 1

        if self.num_total_steps % 100 == 0:
            clipping_percentage = 100.0 * self.num_clipped_steps / self.num_total_steps
            metrics["gradient/clipping_percentage"] = clipping_percentage

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    # ========================================================================
    # 7. LEARNING RATE MONITORING
    # ========================================================================

    def log_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log learning rate information.

        Args:
            optimizer: Optimizer with param groups
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        # Log learning rate from first param group
        if len(optimizer.param_groups) > 0:
            metrics["learning_rate/base_lr"] = optimizer.param_groups[0]["lr"]

        # Log for each param group (if multiple LRs)
        for i, pg in enumerate(optimizer.param_groups):
            metrics[f"learning_rate/param_group_{i}"] = pg["lr"]

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    # ========================================================================
    # 8. HIERARCHY LEVEL ANALYSIS
    # ========================================================================

    def log_hierarchy_feature_stats(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log feature statistics per hierarchy level.

        Args:
            predictions: List of prediction tensors per level
            targets: List of target tensors per level
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        for level in range(len(predictions)):
            pred = predictions[level]  # [B, N_level, D]

            # Feature statistics
            feat_mean = pred.mean().item()
            feat_std = pred.std().item()
            feat_min = pred.min().item()
            feat_max = pred.max().item()

            metrics[f"hierarchy/level{level}_feat_mean"] = feat_mean
            metrics[f"hierarchy/level{level}_feat_std"] = feat_std
            metrics[f"hierarchy/level{level}_feat_range"] = feat_max - feat_min

            # Norm statistics (magnitude of embeddings)
            feat_norm = torch.norm(pred, dim=-1).mean().item()
            metrics[f"hierarchy/level{level}_feat_norm"] = feat_norm

            # Patch count at this level
            num_patches_level = pred.shape[1]
            metrics[f"hierarchy/level{level}_num_patches"] = num_patches_level

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    # ========================================================================
    # 9. TRAINING STABILITY MONITORING
    # ========================================================================

    def log_training_stability(
        self,
        current_loss: float,
        loss_history: List[float],
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log training stability metrics.

        Args:
            current_loss: Current loss value
            loss_history: List of recent loss values (e.g., last 50)
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        if len(loss_history) >= 2:
            # Loss smoothness (variance of recent losses)
            loss_smoothness = (
                np.std(loss_history[-50:]) if len(loss_history) >= 50 else np.std(loss_history)
            )
            metrics["stability/loss_smoothness"] = loss_smoothness

            # Loss trend (slope of recent losses)
            if len(loss_history) >= 5:
                recent_losses = np.array(loss_history[-20:])
                x = np.arange(len(recent_losses))
                coeffs = np.polyfit(x, recent_losses, 1)
                loss_trend_slope = coeffs[0]
                metrics["stability/loss_trend_slope"] = loss_trend_slope

            # Loss spike detection
            max_loss = max(loss_history)
            min_loss = min(loss_history)
            loss_spike_ratio = max_loss / (min_loss + 1e-8)
            metrics["stability/loss_spike_ratio"] = loss_spike_ratio

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics

    # ========================================================================
    # 10. PERFORMANCE AND EFFICIENCY
    # ========================================================================

    def log_performance_metrics(
        self,
        batch_size: int,
        forward_time: float,
        backward_time: float,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log training performance metrics.

        Args:
            batch_size: Batch size
            forward_time: Time for forward pass (seconds)
            backward_time: Time for backward pass (seconds)
            global_step: Current training step

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}

        # Timing
        metrics["performance/forward_time_ms"] = forward_time * 1000
        metrics["performance/backward_time_ms"] = backward_time * 1000
        metrics["performance/total_time_ms"] = (forward_time + backward_time) * 1000

        # Throughput
        total_time = forward_time + backward_time
        if total_time > 0:
            samples_per_second = batch_size / total_time
            metrics["performance/samples_per_second"] = samples_per_second

        # Forward/backward ratio
        if backward_time > 0:
            metrics["performance/forward_backward_ratio"] = forward_time / backward_time

        # Log GPU memory
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9

            metrics["performance/gpu_memory_allocated_gb"] = allocated_gb
            metrics["performance/gpu_memory_reserved_gb"] = reserved_gb

        # Log to metrics logger
        self.metrics_logger.log_metrics(
            metrics,
            step=global_step,
            prefix="train/",
            commit=False,
        )

        return metrics
