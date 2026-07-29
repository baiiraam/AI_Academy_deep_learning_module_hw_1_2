"""
src/optimizers/momentum.py

SGD with Momentum optimizer.
"""

import numpy as np

from src.optimizers.base import Optimizer


class Momentum(Optimizer):
    """
    SGD with Momentum optimizer.

    Update:
        v = momentum * v + grad + weight_decay * θ
        θ = θ - lr * v

    Args:
        params: List of trainable Tensor parameters
        lr: Learning rate
        momentum: Momentum coefficient (default: 0.9)
        weight_decay: L2 regularization coefficient (default: 0.0)
    """

    def __init__(
        self, params: list, lr: float, momentum: float = 0.9, weight_decay: float = 0.0
    ):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        # Initialize velocity for each parameter
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            # Apply weight decay
            grad = p.grad + self.weight_decay * p.data

            # Update velocity
            self.velocities[i] = self.momentum * self.velocities[i] + grad

            # Update parameters
            p.data -= self.lr * self.velocities[i]
