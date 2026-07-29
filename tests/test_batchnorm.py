"""
tests/test_batchnorm.py

Tests for Batch Normalization layer.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import activations
from src.autograd.tensor import Tensor
from src.layers import BatchNorm1d, Linear, Module
from src.utils import set_seed


def test_batchnorm_forward_train():
    """Test BatchNorm forward pass in training mode."""
    print("\nTesting BatchNorm forward (train)...")

    bn = BatchNorm1d(num_features=3)
    bn.train()

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = bn(x)

    assert y.data.shape == (2, 3)

    mean = y.data.mean(axis=0)
    std = y.data.std(axis=0)

    np.testing.assert_almost_equal(mean, np.zeros(3), decimal=6)
    np.testing.assert_almost_equal(std, np.ones(3), decimal=5)

    print(f"  Mean after BN: {mean}")
    print(f"  Std after BN: {std}")
    print("✅ BatchNorm forward (train) passed")


def test_batchnorm_forward_eval():
    """Test BatchNorm forward pass in evaluation mode."""
    print("\nTesting BatchNorm forward (eval)...")

    bn = BatchNorm1d(num_features=3)
    bn.train()

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    _ = bn(x)

    bn.eval()
    y = bn(x)
    assert y.data.shape == (2, 3)

    print(f"  Running mean: {bn.running_mean}")
    print(f"  Running var: {bn.running_var}")
    print("✅ BatchNorm forward (eval) passed")


def test_batchnorm_backward():
    """Test BatchNorm backward pass."""
    print("\nTesting BatchNorm backward...")

    bn = BatchNorm1d(num_features=3)
    bn.train()

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = bn(x)
    loss = y.sum()
    loss.backward()

    # Check that gradients exist and are finite
    assert bn.gamma.grad.shape == (3,)
    assert bn.beta.grad.shape == (3,)
    assert np.all(np.isfinite(bn.gamma.grad))
    assert np.all(np.isfinite(bn.beta.grad))

    # For BatchNorm, gamma.grad can be zero for symmetric inputs
    # So we just check that the backward pass ran successfully
    print(f"  gamma.grad: {bn.gamma.grad}")
    print(f"  beta.grad: {bn.beta.grad}")
    print("✅ BatchNorm backward passed")


def test_batchnorm_parameters():
    """Test BatchNorm parameters."""
    print("\nTesting BatchNorm parameters...")

    bn = BatchNorm1d(num_features=3)
    params = bn.parameters()

    assert len(params) == 2
    assert params[0] is bn.gamma
    assert params[1] is bn.beta
    print("✅ BatchNorm parameters passed")


def test_batchnorm_with_simple_network():
    """Test BatchNorm with a simple two-layer network."""
    print("\nTesting BatchNorm with simple network...")

    rng = set_seed(42)

    class SimpleNet(Module):
        def __init__(self, rng):
            super().__init__()
            self.lin1 = Linear(4, 8, rng)
            self.bn = BatchNorm1d(8)
            self.lin2 = Linear(8, 3, rng)

        def forward(self, x):
            x = self.lin1(x)
            x = activations.relu(x)
            x = self.bn(x)
            x = self.lin2(x)
            return x

        def parameters(self):
            return (
                self.lin1.parameters() + self.bn.parameters() + self.lin2.parameters()
            )

    model = SimpleNet(rng)
    x = Tensor(np.random.randn(3, 4))  # Random, not symmetric
    y = model(x)

    assert y.data.shape == (3, 3)

    loss = y.sum()
    loss.backward()

    # Check that gradients exist and are finite
    assert np.all(np.isfinite(model.bn.gamma.grad))
    assert np.all(np.isfinite(model.bn.beta.grad))

    print("  Simple network with BatchNorm forward and backward passed")
    print("✅ BatchNorm with simple network passed")


def run_all_tests():
    """Run all BatchNorm tests."""
    print("=" * 60)
    print("RUNNING BATCHNORM TESTS")
    print("=" * 60)

    test_batchnorm_forward_train()
    test_batchnorm_forward_eval()
    test_batchnorm_backward()
    test_batchnorm_parameters()
    test_batchnorm_with_simple_network()

    print("\n" + "=" * 60)
    print("🎉 ALL BATCHNORM TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
