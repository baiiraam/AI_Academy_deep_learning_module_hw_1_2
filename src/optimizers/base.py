"""
src/optimizers/base.py

Abstract base class for all optimizers.
"""

from abc import ABC, abstractmethod

from src.autograd.tensor import Tensor


class Optimizer(ABC):
    """
    Base class for all optimizers.

    Args:
        params: List of trainable Tensor parameters
        lr: Learning rate
        weight_decay: L2 regularization coefficient
    """

    def __init__(self, params: list[Tensor], lr: float, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def step(self) -> None:
        """Update parameters using their gradients."""

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero."""
        for p in self.params:
            p.zero_grad()
