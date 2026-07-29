"""
src/models/perceptron.py

MLP model built from Linear layers and activations.
"""

import numpy as np

from src import activations
from src.autograd.tensor import Tensor
from src.layers import Dropout, Linear, Module


class MLP(Module):
    """
    Multi-Layer Perceptron.

    Args:
        sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        rng: NumPy random generator
        dropout: Dropout probability (0 = no dropout)

    Example:
        mlp = MLP([64, 128, 32, 10], rng, dropout=0.2)
        # Input: 64, Hidden1: 128, Hidden2: 32, Output: 10
    """

    def __init__(
        self, sizes: list[int], rng: np.random.Generator, dropout: float = 0.0
    ):
        super().__init__()

        self.sizes = sizes
        self.dropout_prob = dropout
        self.layers = []

        # Build layers: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> ...
        for i in range(len(sizes) - 1):
            # Linear layer
            lin = Linear(sizes[i], sizes[i + 1], rng)
            self.layers.append(lin)

            # Add dropout after each layer EXCEPT the output
            if i < len(sizes) - 2 and dropout > 0.0:
                self.layers.append(Dropout(dropout, rng))

        # Store references to layers with parameters for parameters()
        self._linear_layers = [l for l in self.layers if isinstance(l, Linear)]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Input Tensor of shape (N, input_dim)

        Returns:
            Tensor: Logits of shape (N, output_dim) — no activation on output
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU after EVERY Linear layer EXCEPT the last one
            # The last layer is Linear (output layer) — no ReLU
            if isinstance(layer, Linear) and i < len(self.layers) - 1:
                x = activations.relu(x)
        return x

    def parameters(self) -> list:
        """Collect all trainable parameters from Linear layers."""
        params = []
        for layer in self._linear_layers:
            params.extend(layer.parameters())
        return params

    def train(self) -> None:
        """Set model to training mode (enables dropout)."""
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        """Set model to evaluation mode (disables dropout)."""
        for layer in self.layers:
            layer.eval()
