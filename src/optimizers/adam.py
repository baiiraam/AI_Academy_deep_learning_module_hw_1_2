"""
src/optimizers/adam.py

Adam optimizer with bias correction.
"""

import numpy as np

from src.optimizers.base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer with bias correction.

    Update:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        θ = θ - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        params: List of trainable Tensor parameters
        lr: Learning rate (default: 1e-3)
        betas: Tuple of (beta1, beta2) (default: (0.9, 0.999))
        eps: Epsilon for numerical stability (default: 1e-8)
        weight_decay: L2 regularization coefficient (default: 0.0)
    """

    def __init__(
        self,
        params: list,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps

        # Initialize moments for each parameter
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0  # Step counter for bias correction

    def step(self) -> None:
        self.t += 1

        for i, p in enumerate(self.params):
            # Apply weight decay
            grad = p.grad + self.weight_decay * p.data

            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
