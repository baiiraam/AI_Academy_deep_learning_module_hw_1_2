"""
tests/test_losses.py

Unit tests for loss functions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor
from src.losses import softmax_cross_entropy


def test_softmax_cross_entropy_forward():
    """Test softmax cross-entropy forward pass."""
    print("\nTesting softmax cross-entropy forward...")

    logits = Tensor(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    y = np.array([2, 1])

    loss = softmax_cross_entropy(logits, y)

    # Loss should be a scalar (0-dimensional)
    assert loss.data.shape == ()
    loss_value = float(loss.data)
    print(f"  Loss value: {loss_value:.4f}")
    assert np.isfinite(loss_value)
    print("✅ softmax cross-entropy forward passed")


def test_softmax_cross_entropy_gradient():
    """Test softmax cross-entropy gradient."""
    print("\nTesting softmax cross-entropy gradient...")

    logits = Tensor(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    y = np.array([2, 1])

    loss = softmax_cross_entropy(logits, y)
    loss.backward()

    # Manual computation
    N = 2
    max_vals = logits.data.max(axis=1, keepdims=True)
    stable_logits = logits.data - max_vals
    exp_logits = np.exp(stable_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0

    expected_grad = (probs - onehot) / N

    np.testing.assert_almost_equal(logits.grad, expected_grad, decimal=7)
    print(f"  Gradient shape: {logits.grad.shape}")
    print(f"  Gradient sum: {logits.grad.sum():.6f}")
    print("✅ softmax cross-entropy gradient passed")


def test_stable_softmax():
    """Test that softmax is numerically stable for large values."""
    print("\nTesting stable softmax...")

    logits = Tensor(np.array([[1000.0, 1001.0, 1002.0]]))
    y = np.array([2])

    loss = softmax_cross_entropy(logits, y)

    loss_value = float(loss.data)
    assert np.isfinite(loss_value)
    print(f"  Loss with large logits: {loss_value:.4f}")
    print("✅ stable softmax passed")


def test_gradient_via_finite_difference():
    """Test gradient using finite difference approximation."""
    print("\nTesting gradient via finite difference...")

    logits_data = np.array([[0.5, 1.5, 2.5], [0.1, 0.2, 0.3]])
    y = np.array([2, 0])

    def loss_fn(logits_tensor):
        return softmax_cross_entropy(logits_tensor, y)

    # Create Tensor and compute gradient
    logits = Tensor(logits_data.copy())
    loss = loss_fn(logits)
    loss.backward()
    analytical_grad = logits.grad.copy()

    # Finite difference
    h = 1e-5
    numerical_grad = np.zeros_like(logits_data)

    for i in range(logits_data.size):
        row = i // logits_data.shape[1]
        col = i % logits_data.shape[1]

        # Forward perturbation
        logits_plus = Tensor(logits_data.copy())
        logits_plus.data[row, col] += h
        loss_plus = float(loss_fn(logits_plus).data)

        # Backward perturbation
        logits_minus = Tensor(logits_data.copy())
        logits_minus.data[row, col] -= h
        loss_minus = float(loss_fn(logits_minus).data)

        numerical_grad[row, col] = (loss_plus - loss_minus) / (2 * h)

    # Compare — fix: use np.maximum with 2 args, then compare with 1.0
    denom = np.maximum(np.abs(analytical_grad), np.abs(numerical_grad))
    denom = np.maximum(denom, 1.0)  # Ensure denominator is at least 1
    rel_error = np.max(np.abs(analytical_grad - numerical_grad) / denom)

    print(f"  Relative error: {rel_error:.2e}")
    assert rel_error < 1e-6, f"Relative error too large: {rel_error}"
    print("✅ gradient via finite difference passed")


def run_all_tests():
    """Run all loss tests."""
    print("=" * 60)
    print("RUNNING LOSS TESTS")
    print("=" * 60)

    test_softmax_cross_entropy_forward()
    test_softmax_cross_entropy_gradient()
    test_stable_softmax()
    test_gradient_via_finite_difference()

    print("\n" + "=" * 60)
    print("🎉 ALL LOSS TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
