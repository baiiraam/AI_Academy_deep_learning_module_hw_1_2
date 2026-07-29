"""
src/utils.py

Utility functions for training, evaluation, and gradient checking.
"""

import numpy as np

from src.autograd.tensor import Tensor
from src.losses import softmax_cross_entropy


def set_seed(student_id: int) -> np.random.Generator:
    """
    Create a seeded random number generator.

    Args:
        student_id: Your Academy student ID (used as seed)

    Returns:
        np.random.Generator: Seeded generator
    """
    return np.random.default_rng(student_id)


def gradient_check(f, x: Tensor, h: float = 1e-5) -> float:
    """
    Perform central finite difference gradient check.

    This is the function you use in Part B3 to verify your autograd engine.
    It compares analytical gradients (from .backward()) with numerical
    gradients (finite differences).

    Args:
        f: A function that takes x and returns a scalar Tensor
        x: The Tensor to check gradients for
        h: Step size for finite difference (default: 1e-5)

    Returns:
        float: The relative error between analytical and numerical gradients
               Should be ≤ 1e-4 for a correct implementation.
    """
    # Forward pass to compute analytical gradient
    loss = f(x)
    x.zero_grad()  # Reset any existing gradients
    loss.backward()
    analytical_grad = x.grad.copy()

    # Central finite difference
    numerical_grad = np.zeros_like(x.data)
    original_data = x.data.copy()

    for i in range(x.data.size):
        # Create perturbation vector
        pert = np.zeros_like(x.data)
        pert.flat[i] = h

        # f(x + h)
        x.data = original_data + pert
        loss_plus = float(f(x).data.item())  # ← FIXED

        # f(x - h)
        x.data = original_data - pert
        loss_minus = float(f(x).data.item())  # ← FIXED

        # Central difference
        numerical_grad.flat[i] = (loss_plus - loss_minus) / (2 * h)

    # Restore original data
    x.data = original_data

    # Compute relative error
    # Use max(|analytical|, |numerical|, 1) for numerical stability
    denom = np.maximum(np.abs(analytical_grad), np.abs(numerical_grad))
    denom = np.maximum(denom, 1.0)
    rel_error = np.max(np.abs(analytical_grad - numerical_grad) / denom)

    return rel_error


def evaluate(
    model, X: np.ndarray, y: np.ndarray, batch_size: int = 128
) -> tuple[float, float]:
    """
    Evaluate model on data.

    Args:
        model: The MLP model
        X: Input data of shape (N, D)
        y: Labels of shape (N,)
        batch_size: Batch size for evaluation

    Returns:
        tuple: (loss, accuracy)
    """
    model.eval()  # Disable dropout

    N = len(X)
    total_loss = 0.0
    total_correct = 0

    for i in range(0, N, batch_size):
        X_batch = X[i : i + batch_size]
        y_batch = y[i : i + batch_size]

        # Convert to Tensor
        X_tensor = Tensor(X_batch)

        # Forward pass
        logits = model(X_tensor)

        # Loss
        loss = softmax_cross_entropy(logits, y_batch)
        total_loss += float(loss.data) * len(X_batch)

        # Accuracy
        preds = np.argmax(logits.data, axis=1)
        total_correct += np.sum(preds == y_batch)

    avg_loss = total_loss / N
    accuracy = total_correct / N

    return avg_loss, accuracy


def train_model(
    model,
    optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    patience: int = 10,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Train the model with mini-batch SGD and early stopping.

    Args:
        model: The MLP model
        optimizer: The optimizer (SGD, Momentum, or Adam)
        X_train: Training data (N, D)
        y_train: Training labels (N,)
        X_val: Validation data
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        patience: Early stopping patience (epochs with no improvement)
        verbose: Print progress logs

    Returns:
        dict: Training history with keys:
            - 'train_loss': list of train losses per epoch
            - 'train_acc': list of train accuracies per epoch
            - 'val_loss': list of validation losses per epoch
            - 'val_acc': list of validation accuracies per epoch
    """
    n_samples = len(X_train)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    best_model_state: list[np.ndarray] | None = None
    patience_counter = 0

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_losses = []

        # Mini-batch training
        model.train()  # Enable dropout
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            # Forward pass
            X_tensor = Tensor(X_batch)
            logits = model(X_tensor)
            loss = softmax_cross_entropy(logits, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update parameters
            optimizer.step()

            epoch_losses.append(float(loss.data))

        # Evaluate at end of epoch
        train_loss, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, X_val, y_val)

        # Store history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model state (copy all parameter data)
            best_model_state = [p.data.copy() for p in model.parameters()]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                # Restore best model (only if we have a best state)
                if best_model_state is not None:
                    for p, state in zip(model.parameters(), best_model_state):
                        p.data = state.copy()
                break

    return history
