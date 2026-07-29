"""
src/losses.py

Loss functions.
"""

import numpy as np

from src.autograd.tensor import Tensor


def softmax_cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    """
    Numerically stable softmax cross-entropy loss.

    Args:
        logits: Tensor of shape (N, C) — raw logits
        y: numpy array of shape (N,) — integer labels (0 to C-1)

    Returns:
        Tensor: scalar loss value (mean over batch)

    Gradient: ∂L/∂logits = (softmax - onehot) / N
    """
    N = logits.data.shape[0]

    # Numerical stability: subtract row max
    max_vals = logits.data.max(axis=1, keepdims=True)
    stable_logits = logits.data - max_vals

    # Softmax
    exp_logits = np.exp(stable_logits)
    sum_exp = exp_logits.sum(axis=1, keepdims=True)
    probs = exp_logits / sum_exp

    # One-hot encoding
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0

    # Cross-entropy loss (mean over batch) — ensure scalar
    loss_val = float(-np.sum(onehot * np.log(probs + 1e-15)) / N)

    # Create loss Tensor with scalar data
    loss = Tensor(np.array(loss_val, dtype=np.float64), _name="loss")

    # Custom backward with the gradient we derived
    def _backward():
        # ∂L/∂logits = (probs - onehot) / N
        grad = (probs - onehot) / N
        logits.grad += grad

    loss._backward = _backward
    loss._prev = {logits}

    return loss
