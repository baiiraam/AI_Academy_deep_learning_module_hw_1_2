"""
tests/test_schedule.py

Tests for learning rate schedulers.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor
from src.optimizers.schedule import CosineAnnealingWithWarmup, StepDecay
from src.optimizers.sgd import SGD


def test_step_decay():
    """Test StepDecay scheduler."""
    print("\nTesting StepDecay...")

    # Simple parameter
    p = Tensor(2.0)
    p.grad = np.array(1.0)
    opt = SGD([p], lr=0.1)
    scheduler = StepDecay(opt, step_size=5, decay_factor=0.5, lr_min=0.001)

    initial_lr = scheduler.get_lr()
    assert initial_lr == 0.1

    # After 4 steps, lr should still be 0.1
    for _ in range(4):
        scheduler.step()
    assert scheduler.get_lr() == 0.1

    # After 5th step, lr should be 0.05
    scheduler.step()
    assert scheduler.get_lr() == 0.05

    # After 5 more steps, lr should be 0.025
    for _ in range(5):
        scheduler.step()
    assert scheduler.get_lr() == 0.025

    # After many steps, should hit lr_min
    for _ in range(50):
        scheduler.step()
    assert scheduler.get_lr() == 0.001

    print(f"  StepDecay final LR: {scheduler.get_lr():.4f}")
    print("✅ StepDecay passed")


def test_cosine_warmup_full():
    """Test CosineAnnealingWithWarmup through all phases."""
    print("\nTesting CosineAnnealingWithWarmup full cycle...")

    p = Tensor(2.0)
    p.grad = np.array(1.0)
    opt = SGD([p], lr=0.1)
    scheduler = CosineAnnealingWithWarmup(
        opt, lr_max=0.1, lr_min=0.001, warmup_epochs=5, total_epochs=20
    )

    # Check warmup phase
    for epoch in range(1, 6):
        scheduler.step()
        expected_lr = 0.1 * (epoch / 5)
        np.testing.assert_almost_equal(scheduler.get_lr(), expected_lr, decimal=6)
        print(f"  Warmup epoch {epoch}: LR = {scheduler.get_lr():.4f}")

    # Check cosine decay phase
    for epoch in range(6, 21):
        scheduler.step()
        print(f"  Decay epoch {epoch}: LR = {scheduler.get_lr():.6f}")

    # Final LR should be near lr_min
    assert scheduler.get_lr() >= 0.00099
    print("✅ CosineAnnealingWithWarmup full cycle passed")


def test_cosine_warmup_clamp():
    """Test that CosineAnnealingWithWarmup clamps progress correctly."""
    print("\nTesting CosineAnnealingWithWarmup clamp...")

    p = Tensor(2.0)
    p.grad = np.array(1.0)
    opt = SGD([p], lr=0.1)
    scheduler = CosineAnnealingWithWarmup(
        opt, lr_max=0.1, lr_min=0.001, warmup_epochs=5, total_epochs=10
    )

    # Run past total_epochs
    for _ in range(20):
        scheduler.step()

    # Should not go below lr_min
    assert scheduler.get_lr() >= 0.00099
    print(f"  Final LR (clamped): {scheduler.get_lr():.6f}")
    print("✅ CosineAnnealingWithWarmup clamp passed")


def run_all_tests():
    """Run all scheduler tests."""
    print("=" * 60)
    print("RUNNING SCHEDULER TESTS")
    print("=" * 60)

    test_step_decay()
    test_cosine_warmup_full()
    test_cosine_warmup_clamp()

    print("\n" + "=" * 60)
    print("🎉 ALL SCHEDULER TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
