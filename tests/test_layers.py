"""
tests/test_layers.py

Unit tests for layers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor
from src.layers import Dropout, Linear


def test_linear_forward():
    """Test Linear forward pass."""
    print("\nTesting Linear forward...")
    rng = np.random.default_rng(42)

    layer = Linear(3, 2, rng)
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    y = layer(x)

    # y = x @ W + b, shape should be (1, 2)
    assert y.data.shape == (1, 2)
    print("✅ Linear forward passed")


def test_linear_parameters():
    """Test Linear parameters."""
    print("\nTesting Linear parameters...")
    rng = np.random.default_rng(42)

    layer = Linear(3, 2, rng)
    params = layer.parameters()

    assert len(params) == 2
    assert params[0] is layer.W
    assert params[1] is layer.b
    print("✅ Linear parameters passed")


def test_linear_backward():
    """Test Linear backward through a loss."""
    print("\nTesting Linear backward...")
    rng = np.random.default_rng(42)

    layer = Linear(3, 2, rng)
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    y = layer(x)

    # Scalar loss: sum of outputs
    loss = y.sum()
    loss.backward()

    # Gradients should be non-zero
    assert np.any(layer.W.grad != 0)
    assert np.any(layer.b.grad != 0)
    print("✅ Linear backward passed")


def test_dropout_train():
    """Test Dropout in training mode."""
    print("\nTesting Dropout train mode...")
    rng = np.random.default_rng(42)

    dropout = Dropout(0.5, rng)
    dropout.train()

    x = Tensor(np.ones((10, 5)))
    y = dropout(x)

    # Some values should be zero, some should be 2 (scaled by 1/(1-0.5)=2)
    assert np.any(y.data == 0)
    assert np.any(y.data == 2.0)
    print("✅ Dropout train passed")


def test_dropout_eval():
    """Test Dropout in evaluation mode."""
    print("\nTesting Dropout eval mode...")
    rng = np.random.default_rng(42)

    dropout = Dropout(0.5, rng)
    dropout.eval()

    x = Tensor(np.ones((10, 5)))
    y = dropout(x)

    # Should be identity (no changes)
    np.testing.assert_almost_equal(y.data, x.data)
    print("✅ Dropout eval passed")


def run_all_tests():
    """Run all layer tests."""
    print("=" * 60)
    print("RUNNING LAYER TESTS")
    print("=" * 60)

    test_linear_forward()
    test_linear_parameters()
    test_linear_backward()
    test_dropout_train()
    test_dropout_eval()

    print("\n" + "=" * 60)
    print("🎉 ALL LAYER TESTS PASSED!")
    print("=" * 60)


def test_module_parameters():
    """Test Module base class parameters."""
    print("\nTesting Module parameters...")
    from src.layers import Module

    m = Module()
    assert m.parameters() == []
    print("✅ Module parameters passed")


def test_module_call():
    """Test Module __call__."""
    print("\nTesting Module __call__...")
    from src.layers import Module

    m = Module()
    try:
        m(Tensor(2.0))
    except NotImplementedError:
        pass  # Expected
    print("✅ Module __call__ passed")


if __name__ == "__main__":
    run_all_tests()
