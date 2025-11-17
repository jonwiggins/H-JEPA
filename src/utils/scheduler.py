"""
Learning rate and EMA coefficient schedulers for H-JEPA training.

Supports:
- Cosine annealing with warmup
- Linear warmup
- EMA momentum scheduling
- Per-hierarchy level schedules
"""

import math
from typing import List, Union


class CosineScheduler:
    """
    Cosine learning rate schedule with linear warmup.

    Learning rate schedule:
    - Linear warmup from 0 to base_lr during warmup_epochs
    - Cosine annealing from base_lr to min_lr for remaining epochs

    Args:
        base_value: Base learning rate value after warmup
        final_value: Final learning rate value at end of training
        epochs: Total number of training epochs
        steps_per_epoch: Number of optimizer steps per epoch
        warmup_epochs: Number of warmup epochs
        start_warmup_value: Initial learning rate at start of warmup
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 0,
        start_warmup_value: float = 0.0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.start_warmup_value = start_warmup_value

        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch

    def __call__(self, step: int) -> float:
        """
        Get learning rate for the given training step.

        Args:
            step: Current training step (global step count)

        Returns:
            Learning rate value for this step
        """
        if step < self.warmup_steps:
            # Linear warmup
            lr = (
                self.start_warmup_value
                + (self.base_value - self.start_warmup_value) * step / self.warmup_steps
            )
        else:
            # Cosine annealing
            step_after_warmup = step - self.warmup_steps
            total_steps_after_warmup = self.total_steps - self.warmup_steps

            lr = self.final_value + (self.base_value - self.final_value) * 0.5 * (
                1.0 + math.cos(math.pi * step_after_warmup / total_steps_after_warmup)
            )

        return lr

    def get_epoch_value(self, epoch: int) -> float:
        """
        Get learning rate for the start of a given epoch.

        Args:
            epoch: Epoch number (0-indexed)

        Returns:
            Learning rate value for this epoch
        """
        step = epoch * self.steps_per_epoch
        return self(step)


class LinearScheduler:
    """
    Linear learning rate schedule with optional warmup.

    Args:
        base_value: Base learning rate value after warmup
        final_value: Final learning rate value at end of training
        epochs: Total number of training epochs
        steps_per_epoch: Number of optimizer steps per epoch
        warmup_epochs: Number of warmup epochs
        start_warmup_value: Initial learning rate at start of warmup
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 0,
        start_warmup_value: float = 0.0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.start_warmup_value = start_warmup_value

        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch

    def __call__(self, step: int) -> float:
        """Get learning rate for the given training step."""
        if step < self.warmup_steps:
            # Linear warmup
            lr = (
                self.start_warmup_value
                + (self.base_value - self.start_warmup_value) * step / self.warmup_steps
            )
        else:
            # Linear decay
            step_after_warmup = step - self.warmup_steps
            total_steps_after_warmup = self.total_steps - self.warmup_steps

            lr = (
                self.base_value
                - (self.base_value - self.final_value)
                * step_after_warmup
                / total_steps_after_warmup
            )

        return lr


class EMAScheduler:
    """
    Exponential Moving Average (EMA) momentum scheduler.

    Schedules the EMA momentum coefficient from start value to end value
    with optional warmup using linear interpolation as per I-JEPA paper.
    Used for updating target encoder in H-JEPA.

    Args:
        base_value: Base EMA momentum after warmup (typically 0.996)
        final_value: Final EMA momentum (typically 1.0)
        epochs: Total number of training epochs
        steps_per_epoch: Number of optimizer steps per epoch
        warmup_epochs: Number of warmup epochs
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs

        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch

    def __call__(self, step: int) -> float:
        """
        Get EMA momentum for the given training step.

        Args:
            step: Current training step

        Returns:
            EMA momentum coefficient for this step
        """
        if step < self.warmup_steps:
            # During warmup, stay at base value
            return self.base_value
        else:
            # Linear schedule from base to final value
            step_after_warmup = step - self.warmup_steps
            total_steps_after_warmup = self.total_steps - self.warmup_steps

            progress = min(1.0, step_after_warmup / total_steps_after_warmup)
            momentum = self.base_value + (self.final_value - self.base_value) * progress

        return momentum


class HierarchicalScheduler:
    """
    Manages separate schedulers for different hierarchy levels.

    Allows different learning rates or schedules for different levels
    of the hierarchical predictor.

    Args:
        schedulers: List of schedulers, one per hierarchy level

    Example:
        >>> # Different learning rates per level
        >>> schedulers = [
        ...     CosineScheduler(1e-3, 1e-6, 100, 1000, warmup_epochs=10),
        ...     CosineScheduler(5e-4, 5e-7, 100, 1000, warmup_epochs=10),
        ...     CosineScheduler(2e-4, 2e-7, 100, 1000, warmup_epochs=10),
        ... ]
        >>> hier_sched = HierarchicalScheduler(schedulers)
        >>> lrs = hier_sched(step=500)  # Returns list of 3 learning rates
    """

    def __init__(self, schedulers: List[Union[CosineScheduler, LinearScheduler, EMAScheduler]]):
        self.schedulers = schedulers
        self.num_levels = len(schedulers)

    def __call__(self, step: int) -> List[float]:
        """
        Get values for all hierarchy levels at the given step.

        Args:
            step: Current training step

        Returns:
            List of values, one per hierarchy level
        """
        return [scheduler(step) for scheduler in self.schedulers]

    def get_level_value(self, step: int, level: int) -> float:
        """
        Get value for a specific hierarchy level.

        Args:
            step: Current training step
            level: Hierarchy level index

        Returns:
            Value for the specified level
        """
        return self.schedulers[level](step)


def create_lr_scheduler(
    optimizer_type: str,
    base_lr: float,
    min_lr: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    schedule_type: str = "cosine",
) -> Union[CosineScheduler, LinearScheduler]:
    """
    Factory function to create learning rate scheduler.

    Args:
        optimizer_type: Type of optimizer (for potential adjustments)
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        epochs: Total epochs
        steps_per_epoch: Steps per epoch
        warmup_epochs: Warmup epochs
        schedule_type: Type of schedule ("cosine" or "linear")

    Returns:
        Learning rate scheduler
    """
    if schedule_type == "cosine":
        return CosineScheduler(
            base_value=base_lr,
            final_value=min_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
            start_warmup_value=0.0,
        )
    elif schedule_type == "linear":
        return LinearScheduler(
            base_value=base_lr,
            final_value=min_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
            start_warmup_value=0.0,
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def create_ema_scheduler(
    base_momentum: float,
    final_momentum: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
) -> EMAScheduler:
    """
    Factory function to create EMA momentum scheduler.

    Args:
        base_momentum: Base EMA momentum (e.g., 0.996)
        final_momentum: Final EMA momentum (e.g., 1.0)
        epochs: Total epochs
        steps_per_epoch: Steps per epoch
        warmup_epochs: Warmup epochs for EMA

    Returns:
        EMA momentum scheduler
    """
    return EMAScheduler(
        base_value=base_momentum,
        final_value=final_momentum,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=warmup_epochs,
    )
