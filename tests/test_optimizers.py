# tests/test_optimizers.py

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd.tensor import Tensor
from src.data import iterate_minibatches, load_data
from src.optimizers.adam import Adam
from src.optimizers.momentum import Momentum
from src.optimizers.schedule import CosineAnnealingWithWarmup
from src.optimizers.sgd import SGD


def test_sgd_step():
    p = Tensor(2.0)
    p.grad = np.array(3.0)
    opt = SGD([p], lr=0.01)
    opt.step()
    assert p.data < 2.0
    print("✅ SGD step passed")


def test_momentum_step():
    p = Tensor(2.0)
    p.grad = np.array(3.0)
    opt = Momentum([p], lr=0.01, momentum=0.9)
    opt.step()
    assert p.data < 2.0
    print("✅ Momentum step passed")


def test_adam_step():
    p = Tensor(2.0)
    p.grad = np.array(3.0)
    opt = Adam([p], lr=0.01)
    opt.step()
    assert p.data < 2.0
    print("✅ Adam step passed")


def test_scheduler_warmup():
    p = Tensor(2.0)
    p.grad = np.array(3.0)
    opt = SGD([p], lr=0.01)
    scheduler = CosineAnnealingWithWarmup(
        opt, lr_max=0.01, lr_min=0.001, warmup_epochs=5, total_epochs=50
    )

    scheduler.step()  # epoch 1
    assert scheduler.get_lr() == 0.002  # 0.01 * (1/5)
    print(f"  LR after 1 step: {scheduler.get_lr():.4f}")

    for _ in range(5):
        scheduler.step()
    # After warmup, should be in cosine decay
    print(f"  LR after warmup: {scheduler.get_lr():.4f}")
    print("✅ Scheduler warmup passed")


def test_data_loading():
    X_train, X_test, _y_train, _y_test, X_val, _y_val = load_data(12345)
    assert len(X_train) > 0
    assert len(X_test) > 0
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print("✅ Data loading passed")


def test_iterate_minibatches():
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, size=100)

    batches = list(iterate_minibatches(X, y, batch_size=32, shuffle=True))
    assert len(batches) > 0
    # First batch should be size 32
    assert batches[0][0].shape[0] == 32
    print("✅ Mini-batch iteration passed")


def run_all_tests():
    print("=" * 60)
    print("RUNNING ADDITIONAL OPTIMIZER TESTS")
    print("=" * 60)

    test_sgd_step()
    test_momentum_step()
    test_adam_step()
    test_scheduler_warmup()
    test_data_loading()
    test_iterate_minibatches()

    print("\n" + "=" * 60)
    print("🎉 ALL ADDITIONAL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
