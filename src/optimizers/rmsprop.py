"""
src/optimizers/rmsprop.py

RMSProp optimizer.
"""

import numpy as np

from src.optimizers.base import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer.

    Update:
        v = beta * v + (1 - beta) * grad^2
        theta = theta - lr * grad / (sqrt(v) + eps)

    Args:
        params: List of trainable Tensor parameters
        lr: Learning rate
        beta: Decay rate for squared gradients (default: 0.9)
        eps: Numerical stability constant (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0.0)
    """

    def __init__(
        self,
        params: list,
        lr: float,
        beta: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay)
        self.beta = beta
        self.eps = eps

        # Initialize running average of squared gradients for each parameter
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Update parameters using RMSProp."""
        for i, p in enumerate(self.params):
            # Apply weight decay
            grad = p.grad + self.weight_decay * p.data

            # Update running average of squared gradients
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (grad**2)

            # Update parameters
            p.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)
