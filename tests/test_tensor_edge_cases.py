"""
tests/test_tensor_edge_cases.py

Tests for edge cases in the Tensor autograd engine.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor


def test_detach():
    """Test detach operation."""
    print("\nTesting detach...")
    x = Tensor(2.0, _name="x")
    y = x.detach()
    assert y.data == 2.0
    assert len(y._prev) == 0
    assert y._name == "detached_x"
    y.data = np.array(5.0)
    assert x.data == 2.0
    print("✅ detach passed")


def test_getitem():
    """Test indexing operation."""
    print("\nTesting __getitem__...")
    x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    y = x[1:3]
    loss = y.sum()
    loss.backward()
    expected = np.array([0.0, 1.0, 1.0, 0.0])
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ __getitem__ passed")


def test_sum_axis():
    """Test sum with axis."""
    print("\nTesting sum(axis=)...")
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = x.sum(axis=0)
    y.backward()
    expected = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ sum(axis=0) passed")


def test_repr():
    """Test Tensor string representation."""
    print("\nTesting __repr__...")
    x = Tensor(np.array([1.0, 2.0, 3.0]), _name="test")
    repr_str = repr(x)
    assert "Tensor" in repr_str
    assert "test" in repr_str
    print("✅ __repr__ passed")


def test_rmul():
    """Test reverse multiplication (int * Tensor)."""
    print("\nTesting __rmul__...")
    x = Tensor(2.0)
    y = 3.0 * x  # Calls __rmul__
    y.backward()
    np.testing.assert_almost_equal(y.data, 6.0)
    np.testing.assert_almost_equal(x.grad, 3.0)
    print("✅ __rmul__ passed")


def test_neg():
    """Test negation."""
    print("\nTesting __neg__...")
    x = Tensor(2.0)
    y = -x
    y.backward()
    np.testing.assert_almost_equal(y.data, -2.0)
    np.testing.assert_almost_equal(x.grad, -1.0)
    print("✅ __neg__ passed")


def test_sub():
    """Test subtraction."""
    print("\nTesting __sub__...")
    a = Tensor(5.0)
    b = Tensor(3.0)
    y = a - b
    y.backward()
    np.testing.assert_almost_equal(y.data, 2.0)
    np.testing.assert_almost_equal(a.grad, 1.0)
    np.testing.assert_almost_equal(b.grad, -1.0)
    print("✅ __sub__ passed")


def test_rsub():
    """Test reverse subtraction."""
    print("\nTesting __rsub__...")
    x = Tensor(5.0)
    y = 10.0 - x  # Calls __rsub__
    y.backward()
    np.testing.assert_almost_equal(y.data, 5.0)
    np.testing.assert_almost_equal(x.grad, -1.0)
    print("✅ __rsub__ passed")


def test_truediv():
    """Test division."""
    print("\nTesting __truediv__...")
    x = Tensor(4.0)
    y = x / Tensor(2.0)
    y.backward()
    np.testing.assert_almost_equal(y.data, 2.0)
    np.testing.assert_almost_equal(x.grad, 0.5)
    print("✅ __truediv__ passed")


def test_pow():
    """Test power operation."""
    print("\nTesting __pow__...")
    x = Tensor(2.0)
    y = x**3
    y.backward()
    np.testing.assert_almost_equal(y.data, 8.0)
    np.testing.assert_almost_equal(x.grad, 12.0)
    print("✅ __pow__ passed")


def test_exp_backward():
    """Test exp backward."""
    print("\nTesting exp backward...")
    x = Tensor(2.0)
    y = x.exp()
    y.backward()
    np.testing.assert_almost_equal(x.grad, np.exp(2.0))
    print("✅ exp backward passed")


def test_tensor_properties():
    """Test Tensor properties: shape, ndim, dtype."""
    print("\nTesting Tensor properties...")
    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    assert x.shape == (2, 3)
    assert x.ndim == 2
    assert x.dtype == np.float64

    s = Tensor(5.0)
    assert s.shape == (1,)
    assert s.ndim == 1
    assert s.dtype == np.float64

    s2 = Tensor(np.array(5.0))
    assert s2.shape == ()
    assert s2.ndim == 0

    print("✅ Tensor properties passed")


def test_rmul_with_broadcast():
    """Test reverse multiplication with broadcasting."""
    print("\nTesting __rmul__ with broadcast...")
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    # 2.0 is a float, so this calls __rmul__ on x
    y = 2.0 * x
    loss = y.sum()
    loss.backward()
    expected = np.array([[2.0, 2.0], [2.0, 2.0]])
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ __rmul__ with broadcast passed")


def test_truediv_scalar():
    """Test division by scalar."""
    print("\nTesting __truediv__ by scalar...")
    x = Tensor(10.0)
    y = x / Tensor(5.0)
    y.backward()
    np.testing.assert_almost_equal(y.data, 2.0)
    np.testing.assert_almost_equal(x.grad, 0.2)
    print("✅ __truediv__ by scalar passed")


def test_pow_with_gradient_flow():
    """Test power with gradient flowing through."""
    print("\nTesting __pow__ with gradient flow...")
    x = Tensor(3.0)
    y = x**2
    z = y * Tensor(2.0)
    z.backward()
    np.testing.assert_almost_equal(x.grad, 12.0)
    print("✅ __pow__ with gradient flow passed")


def test_rmul_specific():
    """Test reverse multiplication specifically."""
    print("\nTesting __rmul__ specific...")
    x = Tensor(2.0)
    y = 3.0 * x
    y.backward()
    np.testing.assert_almost_equal(y.data, 6.0)
    np.testing.assert_almost_equal(x.grad, 3.0)
    print("✅ __rmul__ specific passed")


def test_sub_specific():
    """Test subtraction specifically."""
    print("\nTesting __sub__ specific...")
    a = Tensor(5.0)
    b = Tensor(3.0)
    y = a - b
    y.backward()
    np.testing.assert_almost_equal(y.data, 2.0)
    np.testing.assert_almost_equal(a.grad, 1.0)
    np.testing.assert_almost_equal(b.grad, -1.0)
    print("✅ __sub__ specific passed")


def test_rsub_specific():
    """Test reverse subtraction specifically."""
    print("\nTesting __rsub__ specific...")
    x = Tensor(5.0)
    y = 10.0 - x
    y.backward()
    np.testing.assert_almost_equal(y.data, 5.0)
    np.testing.assert_almost_equal(x.grad, -1.0)
    print("✅ __rsub__ specific passed")


def test_truediv_specific():
    """Test division specifically."""
    print("\nTesting __truediv__ specific...")
    x = Tensor(4.0)
    y = x / Tensor(2.0)
    y.backward()
    np.testing.assert_almost_equal(y.data, 2.0)
    np.testing.assert_almost_equal(x.grad, 0.5)
    print("✅ __truediv__ specific passed")


def test_exp_backward_specific():
    """Test exp backward specifically."""
    print("\nTesting exp backward specific...")
    x = Tensor(2.0)
    y = x.exp()
    y.backward()
    np.testing.assert_almost_equal(x.grad, np.exp(2.0))
    print("✅ exp backward specific passed")


def test_sum_axis_edge_case():
    """Test sum with axis edge case."""
    print("\nTesting sum axis edge case...")
    x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))
    y = x.sum(axis=1)
    y.backward()
    expected = np.ones_like(x.data)
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ sum axis edge case passed")


def test_sum_axis_multiple():
    """Test sum with multiple axes."""
    print("\nTesting sum with multiple axes...")
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    # Sum over all axes one at a time
    y = x.sum(axis=0)
    z = y.sum(axis=0)
    z.backward()
    expected = np.ones_like(x.data)
    np.testing.assert_almost_equal(x.grad, expected)
    print("✅ sum with multiple axes passed")


def test_neg_specific():
    """Test negation specifically."""
    print("\nTesting __neg__ specific...")
    x = Tensor(2.0)
    y = -x
    y.backward()
    np.testing.assert_almost_equal(y.data, -2.0)
    np.testing.assert_almost_equal(x.grad, -1.0)
    print("✅ __neg__ specific passed")


def run_all_tests():
    """Run all edge case tests."""
    print("=" * 60)
    print("RUNNING TENSOR EDGE CASE TESTS")
    print("=" * 60)

    test_detach()
    test_getitem()
    test_sum_axis()
    test_repr()
    test_rmul()
    test_neg()
    test_sub()
    test_rsub()
    test_truediv()
    test_pow()
    test_exp_backward()
    test_tensor_properties()
    test_rmul_with_broadcast()
    test_truediv_scalar()
    test_pow_with_gradient_flow()
    test_rmul_specific()
    test_sub_specific()
    test_rsub_specific()
    test_truediv_specific()
    test_exp_backward_specific()
    test_sum_axis_edge_case()
    test_sum_axis_multiple()
    test_neg_specific()

    print("\n" + "=" * 60)
    print("🎉 ALL TENSOR EDGE CASE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
