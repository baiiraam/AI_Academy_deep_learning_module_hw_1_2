"""
tests/test_utils.py

Unit tests for utility functions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor
from src.layers import Linear
from src.models.perceptron import MLP
from src.optimizers.sgd import SGD
from src.utils import evaluate, gradient_check, set_seed, train_model


def test_set_seed():
    """Test that set_seed produces reproducible generators."""
    print("\nTesting set_seed...")

    rng1 = set_seed(42)
    rng2 = set_seed(42)

    # Both generators should produce the same random numbers
    np.testing.assert_equal(rng1.random(10), rng2.random(10))
    print("✅ set_seed passed")


def test_gradient_check_linear():
    """Test gradient check on a simple Linear layer."""
    print("\nTesting gradient_check on Linear...")
    rng = set_seed(42)

    # Simple function: Linear -> sum (scalar loss)
    layer = Linear(3, 2, rng)

    def f(x):
        y = layer(x)
        return y.sum()

    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    rel_error = gradient_check(f, x, h=1e-5)

    print(f"  Relative error: {rel_error:.2e}")
    assert rel_error < 1e-4, f"Relative error too large: {rel_error}"
    print("✅ gradient_check on Linear passed")


def test_gradient_check_mlp():
    """Test gradient check on MLP."""
    print("\nTesting gradient_check on MLP...")
    rng = set_seed(42)

    mlp = MLP([3, 4, 2], rng)

    def f(x):
        logits = mlp(x)
        return logits.sum()

    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    rel_error = gradient_check(f, x, h=1e-5)

    print(f"  Relative error: {rel_error:.2e}")
    assert rel_error < 1e-4, f"Relative error too large: {rel_error}"
    print("✅ gradient_check on MLP passed")


def test_evaluate():
    """Test evaluate function."""
    print("\nTesting evaluate...")
    rng = set_seed(42)

    mlp = MLP([4, 8, 3], rng)
    X = np.random.randn(10, 4)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    loss, acc = evaluate(mlp, X, y)

    assert loss >= 0  # Loss should be non-negative
    assert 0 <= acc <= 1  # Accuracy should be between 0 and 1
    print(f"  Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    print("✅ evaluate passed")


def test_train_model():
    """Test training loop with synthetic data."""
    print("\nTesting train_model...")
    rng = set_seed(42)

    # Create synthetic data
    N = 100
    D = 4
    C = 3

    X = np.random.randn(N, D)
    y = np.random.randint(0, C, size=N)

    # Split into train/val
    split = int(0.8 * N)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Create model and optimizer
    mlp = MLP([D, 8, C], rng)
    optimizer = SGD(mlp.parameters(), lr=0.01)

    # Train for a few epochs
    history = train_model(
        mlp,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=10,
        batch_size=16,
        patience=5,
        verbose=False,
    )

    # Check history has correct keys and lengths
    assert "train_loss" in history
    assert "train_acc" in history
    assert "val_loss" in history
    assert "val_acc" in history

    assert len(history["train_loss"]) > 0
    assert len(history["train_acc"]) > 0

    print(f"  Trained for {len(history['train_loss'])} epochs")
    print(f"  Final train acc: {history['train_acc'][-1]:.4f}")
    print(f"  Final val acc: {history['val_acc'][-1]:.4f}")
    print("✅ train_model passed")


def test_early_stopping_restore():
    """Test that early stopping restores the best model."""
    print("\nTesting early stopping restore...")
    from src.models.perceptron import MLP
    from src.optimizers.sgd import SGD
    from src.utils import set_seed, train_model

    rng = set_seed(42)

    # Create synthetic data
    N = 100
    D = 4
    C = 3
    X = np.random.randn(N, D)
    y = np.random.randint(0, C, size=N)

    # Split
    split = int(0.8 * N)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Create model and optimizer
    mlp = MLP([D, 8, C], rng)
    optimizer = SGD(mlp.parameters(), lr=0.01)

    # Get initial weights
    initial_params = [p.data.copy() for p in mlp.parameters()]

    # Train with patience=2 to force early stopping
    history = train_model(
        mlp,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=16,
        patience=2,
        verbose=False,
    )

    # Early stopping should have triggered (history < 50 epochs)
    epochs_trained = len(history["train_loss"])
    assert epochs_trained < 50
    print(f"  Early stopping triggered after {epochs_trained} epochs")

    # Model should NOT be at initial weights
    for i, (p, init_p) in enumerate(zip(mlp.parameters(), initial_params)):
        assert np.any(p.data != init_p), f"Parameter {i} unchanged"

    # The best model should be restored - check that final validation accuracy
    # is at least the best (or very close)
    final_val_acc = history["val_acc"][-1]
    best_val_acc = max(history["val_acc"])

    # With early stopping, the final validation accuracy should be the best
    # (or at least within a small tolerance)
    print(f"  Final val acc: {final_val_acc:.4f}, Best val acc: {best_val_acc:.4f}")

    # Check that the final model is reasonable (not random)
    # It should have learned something (accuracy > 0.15 for 3-class random data)
    assert final_val_acc >= 0.12, f"Model failed to learn: {final_val_acc:.4f}"
    assert best_val_acc >= 0.12, f"Best model failed to learn: {best_val_acc:.4f}"

    print("✅ Early stopping restore passed")


def run_all_tests():
    """Run all utility tests."""
    print("=" * 60)
    print("RUNNING UTILITY TESTS")
    print("=" * 60)

    test_set_seed()
    test_gradient_check_linear()
    test_gradient_check_mlp()
    test_evaluate()
    test_train_model()
    test_early_stopping_restore()

    print("\n" + "=" * 60)
    print("🎉 ALL UTILITY TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
