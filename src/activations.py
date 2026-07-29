"""
src/activations.py

Activation functions as Tensor operations.
"""

from src.autograd.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """
    ReLU activation: max(0, x)

    Args:
        x: Input Tensor

    Returns:
        Tensor: ReLU applied element-wise
    """
    return x.relu()


def tanh(x: Tensor) -> Tensor:
    """
    Tanh activation.

    Args:
        x: Input Tensor

    Returns:
        Tensor: tanh applied element-wise
    """
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation: 1/(1 + exp(-x))

    Args:
        x: Input Tensor

    Returns:
        Tensor: sigmoid applied element-wise
    """
    return x.sigmoid()
