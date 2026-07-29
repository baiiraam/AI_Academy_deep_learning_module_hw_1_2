"""
src/data.py

Data loading and batching utilities.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(student_id: int, test_size: float = 0.2, val_size: float = 0.1):
    """
    Load the digits dataset and split into train/val/test.

    Args:
        student_id: Your student ID (used for reproducible split)
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation

    Returns:
        tuple: (X_train, X_test, y_train, y_test, X_val, y_val)
    """
    # Load digits
    digits = load_digits()
    X, y = digits.data, digits.target

    # Normalize to [0, 1]
    X = X / 16.0

    # Split with seed for reproducibility
    np.random.default_rng(student_id)

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(student_id % 2**31)
    )

    # Second split: train vs val
    val_size_actual = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size_actual,
        random_state=int(student_id % 2**31),
    )

    return X_train, X_test, y_train, y_test, X_val, y_val


def iterate_minibatches(X, y, batch_size, shuffle=True):
    """
    Generate mini-batches from data.

    Args:
        X: Input data (N, D)
        y: Labels (N,)
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Yields:
        tuple: (X_batch, y_batch)
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield X[batch_indices], y[batch_indices]
