"""
tests/test_activations.py

Unit tests for activation functions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations import relu, sigmoid, tanh
from src.autograd.tensor import Tensor


def test_relu_forward():
    """Test ReLU forward pass."""
    print("\nTesting ReLU forward...")
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    y = relu(x)

    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_almost_equal(y.data, expected)
    print("✅ ReLU forward passed")


def test_relu_backward():
    """Test ReLU backward pass."""
    print("\nTesting ReLU backward...")
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    y = relu(x)
    loss = y.sum()
    loss.backward()

    expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])  # ReLU derivative
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ ReLU backward passed")


def test_tanh_forward():
    """Test tanh forward pass."""
    print("\nTesting tanh forward...")
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = tanh(x)

    expected = np.tanh(np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_almost_equal(y.data, expected)
    print("✅ tanh forward passed")


def test_tanh_backward():
    """Test tanh backward pass."""
    print("\nTesting tanh backward...")
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = tanh(x)
    loss = y.sum()
    loss.backward()

    # ∂tanh/∂x = 1 - tanh²(x)
    expected = 1 - np.tanh(np.array([-1.0, 0.0, 1.0])) ** 2
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ tanh backward passed")


def test_sigmoid_forward():
    """Test sigmoid forward pass."""
    print("\nTesting sigmoid forward...")
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = sigmoid(x)

    expected = 1 / (1 + np.exp(-np.array([-1.0, 0.0, 1.0])))
    np.testing.assert_almost_equal(y.data, expected)
    print("✅ sigmoid forward passed")


def test_sigmoid_backward():
    """Test sigmoid backward pass."""
    print("\nTesting sigmoid backward...")
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = sigmoid(x)
    loss = y.sum()
    loss.backward()

    # ∂sigmoid/∂x = sigmoid(x) * (1 - sigmoid(x))
    s = 1 / (1 + np.exp(-np.array([-1.0, 0.0, 1.0])))
    expected = s * (1 - s)
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ sigmoid backward passed")


def run_all_tests():
    """Run all activation tests."""
    print("=" * 60)
    print("RUNNING ACTIVATION TESTS")
    print("=" * 60)

    test_relu_forward()
    test_relu_backward()
    test_tanh_forward()
    test_tanh_backward()
    test_sigmoid_forward()
    test_sigmoid_backward()

    print("\n" + "=" * 60)
    print("🎉 ALL ACTIVATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
