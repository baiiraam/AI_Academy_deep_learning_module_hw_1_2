"""
experiments/compare_optimizers.py

Compare optimizers (SGD, Momentum, Adam, RMSProp, RMSProp+Schedule) on the same data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from src.autograd.tensor import Tensor
from src.data import load_data
from src.losses import softmax_cross_entropy
from src.models.perceptron import MLP
from src.optimizers.adam import Adam
from src.optimizers.momentum import Momentum
from src.optimizers.rmsprop import RMSProp
from src.optimizers.schedule import CosineAnnealingWithWarmup
from src.optimizers.sgd import SGD
from src.utils import evaluate, set_seed, train_model


def train_with_scheduler(
    model,
    optimizer,
    scheduler,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    verbose=True,
):
    """Train model with a learning rate scheduler."""
    n_samples = len(X_train)
    val_accs = []

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Training
        model.train()
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            X_tensor = Tensor(X_batch)
            logits = model(X_tensor)
            loss = softmax_cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        _, val_acc = evaluate(model, X_val, y_val)
        val_accs.append(val_acc)

        # Step scheduler
        scheduler.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_lr():.6f}"
            )

    return val_accs


def main(student_id: int = 12345):
    """
    Compare optimizers and save comparison plot.
    """
    # Set seed for reproducibility
    rng = set_seed(student_id)

    # Load data (same split for all)
    X_train, X_test, y_train, _y_test, X_val, y_val = load_data(student_id)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Model configuration
    hidden_sizes = [64, 32]
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    sizes = [input_size] + hidden_sizes + [output_size]

    dropout = 0.1
    lr = 0.001
    batch_size = 32
    epochs = 50
    patience = 10

    # Train with each optimizer
    histories = {}

    # --- SGD ---
    print("\nTraining with SGD...")
    model = MLP(sizes, rng, dropout=dropout)
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=1e-4)
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
    histories["SGD"] = history

    # --- Momentum ---
    print("\nTraining with Momentum...")
    model = MLP(sizes, rng, dropout=dropout)
    optimizer = Momentum(model.parameters(), lr=lr, momentum=0.9)
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
    histories["Momentum"] = history

    # --- Adam ---
    print("\nTraining with Adam...")
    model = MLP(sizes, rng, dropout=dropout)
    optimizer = Adam(model.parameters(), lr=lr)
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
    histories["Adam"] = history

    # --- RMSProp ---
    print("\nTraining with RMSProp...")
    model = MLP(sizes, rng, dropout=dropout)
    optimizer = RMSProp(model.parameters(), lr=lr, beta=0.9)
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
    histories["RMSProp"] = history

    # --- RMSProp with Cosine Annealing Schedule ---
    print("\nTraining with RMSProp + Schedule...")
    model = MLP(sizes, rng, dropout=dropout)
    optimizer = RMSProp(model.parameters(), lr=lr, beta=0.9)
    scheduler = CosineAnnealingWithWarmup(
        optimizer, lr_max=lr, lr_min=lr * 0.01, warmup_epochs=5, total_epochs=epochs
    )

    val_accs = train_with_scheduler(
        model,
        optimizer,
        scheduler,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )
    histories["RMSProp+Schedule"] = {"val_acc": val_accs}

    # Create figure
    os.makedirs("figures", exist_ok=True)

    _, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "SGD": "blue",
        "Momentum": "green",
        "Adam": "red",
        "RMSProp": "orange",
        "RMSProp+Schedule": "purple",
    }
    styles = {
        "SGD": "-",
        "Momentum": "--",
        "Adam": "-.",
        "RMSProp": ":",
        "RMSProp+Schedule": "-",
    }

    for name, history in histories.items():
        if "val_acc" in history:
            ax.plot(
                history["val_acc"],
                label=f"{name}",
                color=colors.get(name, "gray"),
                linestyle=styles.get(name, "-"),
                linewidth=2,
            )

    ax.axhline(y=0.95, color="r", linestyle=":", label="Target 0.95", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Optimizer Comparison (Student ID: {student_id})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/optimizer_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✅ Saved: figures/optimizer_comparison.png (with RMSProp)")
    plt.close()

    return histories


if __name__ == "__main__":
    student_id_str = os.environ.get("STUDENT_ID", "12345")
    try:
        student_id = int(student_id_str)
    except ValueError:
        student_id = 12345

    main(student_id)
