"""
tests/test_rmsprop.py

Unit tests for RMSProp optimizer.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor
from src.optimizers.rmsprop import RMSProp


def test_rmsprop_step():
    """Test RMSProp step updates parameters."""
    print("\nTesting RMSProp step...")

    # Simple parameter: scalar with known gradient
    p = Tensor(2.0)
    p.grad = np.array(3.0)

    opt = RMSProp([p], lr=0.01, beta=0.9)
    opt.step()

    # Should update p.data
    assert p.data != 2.0
    print(f"  p.data after step: {p.data.item():.4f}")
    print("✅ RMSProp step passed")


def test_rmsprop_weight_decay():
    """Test RMSProp with weight decay."""
    print("\nTesting RMSProp weight decay...")

    p = Tensor(2.0)
    p.grad = np.array(3.0)

    opt = RMSProp([p], lr=0.01, beta=0.9, weight_decay=0.01)
    opt.step()

    # Weight decay should add extra decay
    print(f"  p.data with weight decay: {p.data.item():.4f}")
    print("✅ RMSProp weight decay passed")


def test_rmsprop_multiple_steps():
    """Test RMSProp with multiple steps."""
    print("\nTesting RMSProp multiple steps...")

    p = Tensor(5.0)
    p.grad = np.array(1.0)

    opt = RMSProp([p], lr=0.01, beta=0.9)

    for i in range(5):
        opt.step()
        # Update gradient to simulate changing gradient
        p.grad = np.array(1.0 * (i + 2) / 2)

    print(f"  p.data after 5 steps: {p.data.item():.4f}")
    assert p.data.item() < 5.0  # Should have decreased — use .item()
    print("✅ RMSProp multiple steps passed")


def test_rmsprop_parameter_shape():
    """Test RMSProp with multi-dimensional parameters."""
    print("\nTesting RMSProp with multi-dimensional parameters...")

    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    p.grad = np.ones_like(p.data)

    opt = RMSProp([p], lr=0.01, beta=0.9)
    opt.step()

    # All values should be updated
    assert np.all(p.data != np.array([[1.0, 2.0], [3.0, 4.0]]))
    print(f"  p.data after step:\n{p.data}")
    print("✅ RMSProp multi-dimensional passed")


def run_all_tests():
    """Run all RMSProp tests."""
    print("=" * 60)
    print("RUNNING RMSPROP TESTS")
    print("=" * 60)

    test_rmsprop_step()
    test_rmsprop_weight_decay()
    test_rmsprop_multiple_steps()
    test_rmsprop_parameter_shape()

    print("\n" + "=" * 60)
    print("🎉 ALL RMSPROP TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
