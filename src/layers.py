"""
src/layers.py

Layer implementations built on Tensor.
"""

import numpy as np

from src.autograd.tensor import Tensor


class Module:
    """Base class for all neural network modules."""

    def parameters(self) -> list:
        """Return list of trainable parameters."""
        return []

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def train(self) -> None:
        """Set module to training mode."""

    def eval(self) -> None:
        """Set module to evaluation mode."""


class Linear(Module):
    """
    Fully connected layer: y = x @ W + b

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rng: NumPy random generator for initialization
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        super().__init__()

        # He initialization (scaled by sqrt(2/in_features))
        scale = np.sqrt(2.0 / in_features)
        W_data = rng.normal(0, scale, size=(in_features, out_features))
        b_data = np.zeros(out_features)

        self.W = Tensor(W_data, _name=f"W_{in_features}x{out_features}")
        self.b = Tensor(b_data, _name=f"b_{out_features}")

        self._trainable = [self.W, self.b]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: x @ W + b"""
        return x @ self.W + self.b

    def parameters(self) -> list:
        return self._trainable


class Dropout(Module):
    """
    Inverted Dropout layer.

    At training time: zeros units with probability p, scales by 1/(1-p)
    At evaluation time: identity (no-op)

    Args:
        p: Dropout probability (probability of zeroing a unit)
        rng: NumPy random generator
    """

    def __init__(self, p: float, rng: np.random.Generator):
        super().__init__()
        self.p = p
        self.rng = rng
        self.training = True
        self._mask = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        # Create mask: 1 with prob (1-p), 0 with prob p
        mask_data = self.rng.binomial(1, 1 - self.p, size=x.data.shape)
        self._mask = Tensor(mask_data, _name="dropout_mask")

        # Scale by 1/(1-p) to keep expected value the same
        # Create scale as a Tensor to satisfy type checker
        scale = Tensor(1.0 / (1.0 - self.p), _name="dropout_scale")
        return x * self._mask * scale

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False
