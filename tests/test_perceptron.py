"""
tests/test_perceptron.py

Unit tests for MLP model.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor
from src.losses import softmax_cross_entropy
from src.models.perceptron import MLP


def test_mlp_forward():
    """Test MLP forward pass."""
    print("\nTesting MLP forward...")
    rng = np.random.default_rng(42)

    mlp = MLP([4, 8, 3], rng)
    x = Tensor(np.random.randn(2, 4))
    y = mlp(x)

    assert y.data.shape == (2, 3)
    print("✅ MLP forward passed")


def test_mlp_parameters():
    """Test MLP parameters collection."""
    print("\nTesting MLP parameters...")
    rng = np.random.default_rng(42)

    mlp = MLP([4, 8, 6, 3], rng)
    params = mlp.parameters()

    # 3 Linear layers → 3 W + 3 b = 6 parameters
    assert len(params) == 6  # 3 layers × (W, b)

    # Check that each parameter is a Tensor
    for p in params:
        assert isinstance(p, Tensor)
    print("✅ MLP parameters passed")


def test_mlp_dropout_toggle():
    """Test MLP train/eval mode toggling."""
    print("\nTesting MLP train/eval...")
    rng = np.random.default_rng(42)

    mlp = MLP([4, 8, 3], rng, dropout=0.5)

    # Check initial state (train by default)
    # All layers should be in train mode
    for layer in mlp.layers:
        if hasattr(layer, "training"):
            assert layer.training is True

    # Switch to eval
    mlp.eval()
    for layer in mlp.layers:
        if hasattr(layer, "training"):
            assert layer.training is False

    # Switch back to train
    mlp.train()
    for layer in mlp.layers:
        if hasattr(layer, "training"):
            assert layer.training is True
    print("✅ MLP train/eval passed")


def test_mlp_backward():
    """Test MLP backward pass through loss."""
    print("\nTesting MLP backward...")
    rng = np.random.default_rng(42)

    mlp = MLP([4, 8, 3], rng)
    x = Tensor(np.random.randn(2, 4))
    y_true = np.array([0, 1])

    logits = mlp(x)
    loss = softmax_cross_entropy(logits, y_true)
    loss.backward()

    # Check that all parameters have gradients
    for param in mlp.parameters():
        assert np.any(param.grad != 0)

    print("✅ MLP backward passed")


def test_mlp_no_dropout():
    """Test MLP with no dropout."""
    print("\nTesting MLP without dropout...")
    rng = np.random.default_rng(42)

    mlp = MLP([4, 8, 3], rng, dropout=0.0)

    x = Tensor(np.random.randn(2, 4))
    y1 = mlp(x)
    y2 = mlp(x)  # Same input, same output (no randomness)

    np.testing.assert_almost_equal(y1.data, y2.data)
    print("✅ MLP without dropout passed")


def run_all_tests():
    """Run all MLP tests."""
    print("=" * 60)
    print("RUNNING MLP TESTS")
    print("=" * 60)

    test_mlp_forward()
    test_mlp_parameters()
    test_mlp_dropout_toggle()
    test_mlp_backward()
    test_mlp_no_dropout()

    print("\n" + "=" * 60)
    print("🎉 ALL MLP TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
