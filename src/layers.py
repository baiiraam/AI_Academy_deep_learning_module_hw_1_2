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


class BatchNorm1d(Module):
    """
    Batch Normalization for 1D data (features dimension is last).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        _name: str | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = Tensor(
            np.ones(num_features), _name=f"{_name}_gamma" if _name else "gamma"
        )
        self.beta = Tensor(
            np.zeros(num_features), _name=f"{_name}_beta" if _name else "beta"
        )

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self._trainable = [self.gamma, self.beta]

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=0, keepdims=True)
            var = x.data.var(axis=0, keepdims=True)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.flatten()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.flatten()

            mean_tensor = Tensor(mean, _name="batch_mean")
            var_tensor = Tensor(var, _name="batch_var")
            one_tensor = Tensor(np.ones((1, self.num_features)), _name="one")
            eps_tensor = Tensor(np.full((1, self.num_features), self.eps), _name="eps")

            x_centered = x - mean_tensor

            inv_std = one_tensor / ((var_tensor + eps_tensor) ** 0.5)

            x_norm = x_centered * inv_std

            # ⚠️ CRITICAL: Multiplying by gamma
            scaled = x_norm * self.gamma
            out = scaled + self.beta
            return out
        else:
            # Evaluation mode
            mean_tensor = Tensor(self.running_mean.reshape(1, -1), _name="running_mean")
            var_tensor = Tensor(self.running_var.reshape(1, -1), _name="running_var")
            one_tensor = Tensor(np.ones((1, self.num_features)), _name="one")
            eps_tensor = Tensor(np.full((1, self.num_features), self.eps), _name="eps")

            x_centered = x - mean_tensor
            inv_std = one_tensor / ((var_tensor + eps_tensor) ** 0.5)
            x_norm = x_centered * inv_std

            out = x_norm * self.gamma + self.beta
            return out

    def parameters(self) -> list:
        return self._trainable

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


class LayerNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, _name: str | None = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.gamma = Tensor(
            np.ones(num_features), _name=f"{_name}_gamma" if _name else "gamma"
        )
        self.beta = Tensor(
            np.zeros(num_features), _name=f"{_name}_beta" if _name else "beta"
        )

        self._trainable = [self.gamma, self.beta]

    def forward(self, x: Tensor) -> Tensor:
        # Compute statistics using .data
        mean = x.data.mean(axis=1, keepdims=True)  # (N, 1)
        var = x.data.var(axis=1, keepdims=True)  # (N, 1)

        mean_tensor = Tensor(mean, _name="layer_mean")
        var_tensor = Tensor(var, _name="layer_var")
        one_tensor = Tensor(np.ones(mean.shape), _name="one")
        eps_tensor = Tensor(np.full(mean.shape, self.eps), _name="eps")

        x_centered = x - mean_tensor
        inv_std = one_tensor / ((var_tensor + eps_tensor) ** 0.5)
        x_norm = x_centered * inv_std

        out = x_norm * self.gamma + self.beta
        return out

    def parameters(self) -> list:
        return self._trainable
