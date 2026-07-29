"""
tests/test_tensor.py

Unit tests for the Tensor autograd engine.
"""

import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor


def test_add_backward():
    """Test addition backward."""
    print("\nTesting addition backward...")
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    c.backward()

    np.testing.assert_almost_equal(a.grad, 1.0)
    np.testing.assert_almost_equal(b.grad, 1.0)
    print("✅ Addition backward passed")


def test_mul_backward():
    """Test multiplication backward."""
    print("\nTesting multiplication backward...")
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    c.backward()

    np.testing.assert_almost_equal(a.grad, 3.0)  # ∂c/∂a = b = 3
    np.testing.assert_almost_equal(b.grad, 2.0)  # ∂c/∂b = a = 2
    print("✅ Multiplication backward passed")


def test_matmul_backward():
    """Test matrix multiplication backward through a scalar loss."""
    print("\nTesting matrix multiplication backward...")
    x = Tensor([[1.0, 2.0]])  # (1, 2)
    W = Tensor([[3.0, 4.0], [5.0, 6.0]])  # (2, 2)
    y = x @ W  # (1, 2)

    # Use a scalar loss (sum of y) to seed gradients correctly
    loss = y.sum()
    loss.backward()

    # ∂loss/∂y = [1, 1]
    # ∂loss/∂x = [1, 1] @ W^T = [1, 1] @ [[3, 5], [4, 6]] = [7, 11]
    expected_x_grad = np.array([[7.0, 11.0]])

    # ∂loss/∂W = x^T @ [1, 1] = [[1], [2]] @ [1, 1] = [[1, 1], [2, 2]]
    expected_W_grad = np.array([[1.0, 1.0], [2.0, 2.0]])

    np.testing.assert_almost_equal(x.grad, expected_x_grad)
    np.testing.assert_almost_equal(W.grad, expected_W_grad)
    print("✅ Matrix multiplication backward passed")


def test_relu_backward():
    """Test ReLU backward."""
    print("\nTesting ReLU backward...")
    x = Tensor(np.array([-1.0, 2.0, -3.0, 4.0]))
    y = x.relu()
    # Use scalar loss
    loss = y.sum()
    loss.backward()

    expected = np.array([0.0, 1.0, 0.0, 1.0])
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ ReLU backward passed")


def test_chain_rule():
    """Test multi-operation chain rule."""
    print("\nTesting chain rule...")
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = Tensor(4.0)

    s = a + b  # s = 5
    y = s * c  # y = 20
    y.backward()

    # ∂y/∂a = ∂y/∂s * ∂s/∂a = c * 1 = 4
    # ∂y/∂b = ∂y/∂s * ∂s/∂b = c * 1 = 4
    # ∂y/∂c = s = 5
    np.testing.assert_almost_equal(a.grad, 4.0)
    np.testing.assert_almost_equal(b.grad, 4.0)
    np.testing.assert_almost_equal(c.grad, 5.0)
    print("✅ Chain rule passed")


def test_sum_backward():
    """Test sum operation backward."""
    print("\nTesting sum backward...")
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = x.sum()
    y.backward()

    np.testing.assert_almost_equal(x.grad, np.array([1.0, 1.0, 1.0]))
    print("✅ Sum backward passed")


def test_auto_grad_accumulation():
    """Test that gradients accumulate when a tensor is used multiple times."""
    print("\nTesting gradient accumulation...")
    x = Tensor(2.0)
    y = x + x  # x used twice
    y.backward()

    # ∂y/∂x = 1 + 1 = 2
    np.testing.assert_almost_equal(x.grad, 2.0)
    print("✅ Gradient accumulation passed")


def test_vector_output_manual_grad():
    """Test vector output with manually provided gradient."""
    print("\nTesting vector output with manual gradient...")
    x = Tensor([[1.0, 2.0]])
    W = Tensor([[3.0, 4.0], [5.0, 6.0]])
    y = x @ W

    # Provide custom gradient [2, 3] — shape must match y's shape (1, 2)
    y.backward(grad=np.array([[2.0, 3.0]]))  # ← Changed to (1, 2)

    # ∂loss/∂x = grad @ W^T = [2, 3] @ [[3, 5], [4, 6]] = [18, 28]
    expected_x_grad = np.array([[18.0, 28.0]])
    np.testing.assert_almost_equal(x.grad, expected_x_grad)
    print("✅ Vector output with manual gradient passed")


def run_all_tests():
    """Run all tensor tests."""
    print("=" * 60)
    print("RUNNING TENSOR ENGINE TESTS")
    print("=" * 60)

    test_add_backward()
    test_mul_backward()
    test_matmul_backward()
    test_relu_backward()
    test_chain_rule()
    test_sum_backward()
    test_auto_grad_accumulation()
    test_vector_output_manual_grad()

    print("\n" + "=" * 60)
    print("🎉 ALL TENSOR TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
