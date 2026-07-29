"""
src/optimizers/sgd.py

Stochastic Gradient Descent optimizer.
"""

from src.optimizers.base import Optimizer


class SGD(Optimizer):
    """
    Plain SGD optimizer.

    Update: θ = θ - lr * (grad + weight_decay * θ)
    """

    def step(self) -> None:
        for p in self.params:
            # Apply weight decay (L2 regularization)
            grad = p.grad + self.weight_decay * p.data
            p.data -= self.lr * grad
