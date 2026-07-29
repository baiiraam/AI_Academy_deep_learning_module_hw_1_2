"""
experiments/train_best.py

Train the best model configuration and generate training curves.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.models.perceptron import MLP
from src.optimizers.adam import Adam
from src.utils import evaluate, set_seed, train_model


def main(student_id: int = 12345):
    """
    Train best model and save training curves.

    Args:
        student_id: Your student ID
    """
    # Set seed
    rng = set_seed(student_id)

    # Load data
    X_train, X_test, y_train, y_test, X_val, y_val = load_data(student_id)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Best configuration
    hidden_sizes = [64, 32]
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    sizes = [input_size] + hidden_sizes + [output_size]

    dropout = 0.2
    lr = 0.001
    weight_decay = 1e-4
    batch_size = 32
    epochs = 100
    patience = 15

    print(f"Model: {sizes}, Dropout: {dropout}, LR: {lr}, WD: {weight_decay}")

    # Create model and optimizer
    model = MLP(sizes, rng, dropout=dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train
    history = train_model(
        model,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=True,
    )

    # Evaluate on test set
    test_loss, test_acc = evaluate(model, X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Create figure directory
    os.makedirs("figures", exist_ok=True)

    # Plot training curves
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves (Student ID: {student_id})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[1]
    ax.plot(history["train_acc"], label="Train Acc", linewidth=2)
    ax.plot(history["val_acc"], label="Val Acc", linewidth=2)
    ax.axhline(y=0.95, color="r", linestyle="--", label="Target 0.95", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Test Acc: {test_acc:.4f} (ID: {student_id})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/training_curves.png", dpi=150, bbox_inches="tight")
    print("\n✅ Saved: figures/training_curves.png")
    plt.close()

    return history, test_acc


if __name__ == "__main__":
    # Get STUDENT_ID from environment, default to '12345'
    student_id_str = os.environ.get("STUDENT_ID", "12345")
    try:
        student_id = int(student_id_str)
    except ValueError:
        student_id = 12345

    main(student_id)
