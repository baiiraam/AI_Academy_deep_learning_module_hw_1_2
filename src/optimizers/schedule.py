"""
src/optimizers/schedule.py

Learning rate schedulers.
"""

import numpy as np


class CosineAnnealingWithWarmup:
    """
    Cosine annealing learning rate schedule with linear warm-up.

    Args:
        optimizer: The optimizer whose learning rate to schedule
        lr_max: Maximum learning rate
        lr_min: Minimum learning rate (end of cosine decay)
        warmup_epochs: Number of epochs for warm-up
        total_epochs: Total number of epochs
    """

    def __init__(
        self,
        optimizer,
        lr_max: float,
        lr_min: float = 0.0,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
    ):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.epoch = 0

    def step(self) -> None:
        """Update learning rate after each epoch."""
        self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            # Linear warm-up: 0 -> lr_max
            lr = self.lr_max * (self.epoch / self.warmup_epochs)
        else:
            # Cosine decay: lr_max -> lr_min
            progress = (self.epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            progress = min(progress, 1.0)  # Clamp to 1.0
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + np.cos(np.pi * progress)
            )

        # Update optimizer's learning rate
        self.optimizer.lr = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr


class StepDecay:
    """
    Step decay learning rate schedule.

    Args:
        optimizer: The optimizer whose learning rate to schedule
        step_size: Number of epochs between decays
        decay_factor: Factor to multiply learning rate by (e.g., 0.5)
        lr_min: Minimum learning rate (stop decaying below this)
    """

    def __init__(
        self,
        optimizer,
        step_size: int = 20,
        decay_factor: float = 0.5,
        lr_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.step_size = step_size
        self.decay_factor = decay_factor
        self.lr_min = lr_min
        self.epoch = 0

    def step(self) -> None:
        """Update learning rate after each epoch."""
        self.epoch += 1

        if self.epoch % self.step_size == 0:
            new_lr = max(self.optimizer.lr * self.decay_factor, self.lr_min)
            self.optimizer.lr = new_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr
