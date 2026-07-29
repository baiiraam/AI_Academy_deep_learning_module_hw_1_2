![Tests](https://img.shields.io/badge/tests-77%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-96%25-yellowgreen)
![Accuracy](https://img.shields.io/badge/accuracy-97.78%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)


# DLE-AI-202 Deep Learning — Homework 1: An MLP from Scratch

> **Autograd engine built using NumPy.**

---

## 📋 Overview

This is my implementation of the **MLP from Scratch** assignment for DLE-AI-202. I built a complete deep learning framework including:

- A custom **autograd engine** (`Tensor` class with automatic differentiation)
- **Neural network layers** (Linear, Dropout, Activations)
- **Loss functions** (Softmax Cross-Entropy)
- **Optimizers** (SGD, Momentum, Adam, RMSProp)
- **Learning rate schedulers** (Cosine Annealing with Warm-up, Step Decay)
- **Training pipeline** with early stopping and gradient checking

The model achieves **97.78% test accuracy** on the digits dataset (97.78%).

---

## 📁 Project Structure

```
hw1_12345/
├── src/
│   ├── autograd/
│   │   └── tensor.py              # Tensor autograd engine
│   ├── models/
│   │   └── perceptron.py          # MLP model
│   ├── optimizers/
│   │   ├── base.py                # Optimizer base class
│   │   ├── sgd.py                 # SGD optimizer
│   │   ├── momentum.py            # SGD + Momentum
│   │   ├── adam.py                # Adam optimizer
│   │   ├── rmsprop.py             # RMSProp optimizer (Bonus)
│   │   └── schedule.py            # Learning rate schedulers (Bonus)
│   ├── __init__.py
│   ├── activations.py             # ReLU, tanh, sigmoid
│   ├── layers.py                  # Linear, Dropout, Module
│   ├── losses.py                  # Softmax Cross-Entropy
│   ├── utils.py                   # Training utilities
│   └── data.py                    # Data loading & batching
├── experiments/
│   ├── train_best.py              # Train best model → training_curves.png
│   └── compare_optimizers.py      # Compare optimizers → optimizer_comparison.png
├── tests/
│   ├── test_activations.py
│   ├── test_layers.py
│   ├── test_losses.py
│   ├── test_optimizers.py
│   ├── test_perceptron.py
│   ├── test_rmsprop.py
│   ├── test_schedule.py
│   ├── test_tensor.py
│   ├── test_tensor_edge_cases.py
│   └── test_utils.py
├── figures/
│   ├── training_curves.png        # Generated training curves
│   └── optimizer_comparison.png   # Generated optimizer comparison
├── run_all.py                     # Main entry point
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
├── README.md                      # This file
└── HW1_Report.pdf                 # Full assignment report
```

---

## 🚀 Getting Started

### Prerequisites

- Python>=3.10

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   With `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Student ID** (replace `12345` with your actual ID):
   ```bash
   export STUDENT_ID=12345   # Linux/macOS
   # or
   set STUDENT_ID=12345      # Windows (cmd)
   $env:STUDENT_ID = 12345   # Windows (PowerShell)
   ```

---

## 🏃 Running the Pipeline

### Run Everything (Tests + Training + Comparison)

```bash
python run_all.py
```

This will:
1. Run all tests
2. Train the best model and generate `training_curves.png`
3. Compare all optimizers and generate `optimizer_comparison.png`
4. Verify the figures were created

### Run Individual Components

**Run all tests:**
```bash
pytest tests/ -v
```

**Run tests with coverage:**
```bash
pytest tests/ --cov=. --cov-report=term-missing
```
instead of ```--cov=.```, you can replace ```.``` with the folder that you want covered.

**Train the best model:**
```bash
python experiments/train_best.py
```

**Compare optimizers:**
```bash
python experiments/compare_optimizers.py
```

---

## 📊 Results

### Test Accuracy

| Metric | Value |
|--------|-------|
| **Final Test Accuracy** | **97.78%** |
| Best Validation Accuracy | 97.22% |
| Early Stopping Epoch | 48 |
| Total Epochs | 100 (stopped at 48) |

### Optimizer Comparison

| Optimizer | Best Val Acc | Epochs to Best | Speed |
|-----------|--------------|----------------|-------|
| **Adam** | 93% | 20 | Fastest |
| **RMSProp + Schedule** | 95% | 50 | Stable |
| **RMSProp** | 94% | 28 | Very good |
| **Momentum** | 91% | 38 | Good |
| **SGD** | 49% | 50 | Slowest |

### Figures
Figures will be generated and saved inside ```figures/``` folder.
| Figure | Description |
|--------|-------------|
| `training_curves.png` | Training and validation loss/accuracy over epochs |
| `optimizer_comparison.png` | Validation accuracy curves for all optimizers |

---

## Tests

### Test Summary

| Category | Test File | Tests |
|----------|-----------|-------|
| Activations | `test_activations.py` | 6 |
| Layers | `test_layers.py` | 7 |
| Losses | `test_losses.py` | 4 |
| Optimizers | `test_optimizers.py` | 6 |
| Perceptron | `test_perceptron.py` | 5 |
| RMSProp | `test_rmsprop.py` | 4 |
| Schedulers | `test_schedule.py` | 3 |
| Tensor | `test_tensor.py` | 8 |
| Tensor Edge Cases | `test_tensor_edge_cases.py` | 23 |
| Utils | `test_utils.py` | 6 |
| **Total** | | **77** |

### Coverage
I ran for the coverage of the whole directory, not specific to ```src/``` folder.
```
Name                                Stmts   Miss  Cover
-----------------------------------------------------------------
src/activations.py                     7      0   100%
src/autograd/tensor.py               175     16    91%
src/data.py                           20      0   100%
src/layers.py                         42      0   100%
src/losses.py                         19      0   100%
src/models/perceptron.py              33      0   100%
src/optimizers/adam.py                19      0   100%
src/optimizers/base.py                12      0   100%
src/optimizers/momentum.py            12      0   100%
src/optimizers/rmsprop.py             13      0   100%
src/optimizers/schedule.py            33      0   100%
src/optimizers/sgd.py                  6      0   100%
src/utils.py                          85      2    98%
-----------------------------------------------------------------
TOTAL                                476     18    96%
```

---

## 🛠️ Built With

| Tool | Purpose |
|------|---------|
| **NumPy** | Numerical computations |
| **Matplotlib** | Plotting training curves |
| **scikit-learn** | Dataset loading and splitting |
| **pytest** | Testing framework |
| **pytest-cov** | Test coverage reporting |
| **Ruff** | Linting |
| **Ty** | Type checking |

---

## 🔬 Gradient Checking

Gradient checking is employed because it is important to check numerical stability of the implementation.
The autograd engine is verified using **central finite difference**:

```
Checked quantity                            | Rel-err
--------------------------------------------|----------
Linear weights ∂L/∂W                        | 2.3 × 10⁻¹⁰
Hidden activation ∂L/∂h                     | 3.4 × 10⁻¹⁰
Loss w.r.t. logits ∂L/∂z₂                   | 2.1 × 10⁻¹⁰
```

All errors are **< 1e-4**, confirming correct gradient computation.

---

## 🎯 Key Features

### 1. Custom Autograd Engine
- Automatic differentiation with reverse-mode backpropagation
- Supports: `+`, `*`, `@`, `sum()`, `relu()`, `exp()`, `log()`, `tanh()`, `sigmoid()`
- Topological sort for correct gradient flow
- Gradient accumulation for shared tensors

### 2. Neural Network Components
- `Linear` layer with He initialization
- `Dropout` with inverted dropout and train/eval modes
- ReLU, tanh, sigmoid activations
- `MLP` assembling all components

### 3. Optimizers
- **SGD** — Plain stochastic gradient descent
- **Momentum** — SGD with momentum (`v = μv + g`)
- **Adam** — Adaptive Moment Estimation with bias correction
- **RMSProp** — Root Mean Square Propagation (Bonus)
- **Learning Rate Schedulers** — Cosine annealing with warm-up, Step decay (Bonus)

### 4. Regularization
- **Dropout** — Inverted dropout with train/eval modes
- **Weight Decay** — L2 regularization in optimizer updates
- **Early Stopping** — Patience-based validation monitoring

### 5. Training Pipeline
- Mini-batch training with shuffling
- Validation-based early stopping
- Automatic figure generation
- Reproducible with seeded RNG

---

## 📝 Report

The full assignment report is available as `HW1_Report.pdf`, containing:

- **Part A** — Written derivations (forward pass, backprop, activations, MLE, optimizers, regularization)
- **Part B** — Implementation details, gradient check, training curves, optimizer comparison
- **Part C** — Code explanations (gradient trace, optimizer contrast, dropout, Adam bias correction)
- **Bonus** — RMSProp + learning rate schedule

---

## 🤖 AI Use Disclosure

**ChatGPT** was used for:
- Explaining Adam bias correction and RMSProp derivations
- Debugging Tensor shape issues in backward passes
- Helping structure and format the report

All code was written, understood, and verified by me. I can explain everything that was submitted.

---

## 👤 Author

| Field | Value |
|-------|-------|
| **Name** | Bayram |
| **Student ID** | placeholder |
| **Section** | placeholder |
| **Course** | DLE-AI-202 Deep Learning (AI Academy) |
| **Date** | July 29, 2026 |

---

## 📚 References

1. Course notes (Weeks 1–2 reading material)
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Ch. 6 (Backpropagation), Ch. 7 (Regularization), Ch. 8 (Optimization)
3. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
4. RmsProp:
https://medium.com/@piyushkashyap045/understanding-rmsprop-a-simple-guide-to-one-of-deep-learnings-powerful-optimizers-403baeed9922
5. SGD with Momentum:
https://medium.com/@piyushkashyap045/understanding-sgd-with-momentum-in-deep-learning-a-beginner-friendly-guide-0252ede605b4
6. Learning Rate Scheduling:
https://machinelearningmastery.com/a-gentle-introduction-to-learning-rate-schedulers/
---

## 📄 License

This project is submitted as coursework for DLE-AI-202 Deep Learning at AI Academy.