"""
src/autograd/tensor.py

A minimal autograd engine for the MLP from scratch assignment.
Each Tensor wraps a NumPy array and records the computation graph.
"""

import numpy as np


class Tensor:
    """
    The core tensor class with automatic differentiation.

    Attributes:
        data (np.ndarray): The actual numeric data
        grad (np.ndarray): Gradient of the loss w.r.t. this tensor
        _prev (Set[Tensor]): Children in the computation graph
        _backward (Callable): Function that propagates gradient to children
        _name (str, optional): Optional name for debugging
    """

    def __init__(
        self,
        data: np.ndarray | list | float,
        _children: tuple["Tensor", ...] = (),
        _name: str | None = None,
        requires_grad: bool = True,
    ):
        # Convert to numpy array with float64 for numerical stability
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=np.float64)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        else:
            data = np.asarray(data, dtype=np.float64)

        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._prev = set(_children)
        self._backward = lambda: None
        self._name = _name
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        name_str = f", name='{self._name}'" if self._name else ""
        return f"Tensor(data={self.data.shape}{name_str})"

    def zero_grad(self) -> None:
        """Reset the gradient to zero."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    # =====================================================================
    #  OPERATIONS
    # =====================================================================

    def __add__(self, other: "Tensor") -> "Tensor":
        """Element-wise addition."""
        other = self._ensure_tensor(other)

        out = Tensor(
            self.data + other.data,
            _children=(self, other),
            _name=f"{self._name}+{other._name}" if self._name else None,
        )

        def _backward():

            if self.requires_grad:
                grad_self = out.grad

                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0, keepdims=False)

                if self.data.size > 1:
                    axes = [
                        i
                        for i in range(len(self.data.shape))
                        if self.data.shape[i] == 1 and grad_self.shape[i] > 1
                    ]
                    if axes:
                        grad_self = grad_self.sum(axis=tuple(axes), keepdims=False)

                if grad_self.shape != self.data.shape:
                    if self.data.size == 1:
                        grad_self = grad_self.sum()
                    else:
                        grad_self = grad_self.reshape(self.data.shape)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = out.grad

                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0, keepdims=False)

                if other.data.size > 1:
                    axes = [
                        i
                        for i in range(len(other.data.shape))
                        if other.data.shape[i] == 1 and grad_other.shape[i] > 1
                    ]
                    if axes:
                        grad_other = grad_other.sum(axis=tuple(axes), keepdims=False)

                if grad_other.shape != other.data.shape:
                    if other.data.size == 1:
                        grad_other = grad_other.sum()
                    else:
                        grad_other = grad_other.reshape(other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        """Reverse addition (for int + Tensor)."""
        return self.__add__(other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Element-wise multiplication."""
        other = self._ensure_tensor(other)

        out = Tensor(
            self.data * other.data,
            _children=(self, other),
            _name=f"{self._name}*{other._name}" if self._name else None,
        )

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * other.data
                # Reduce to match self.data shape
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0, keepdims=False)
                # Handle broadcasting
                if grad_self.shape != self.data.shape:
                    axes = [
                        i
                        for i in range(grad_self.ndim)
                        if i < len(self.data.shape) and self.data.shape[i] == 1
                    ]
                    if axes:
                        grad_self = grad_self.sum(axis=tuple(axes), keepdims=False)
                # Final reshape
                if grad_self.shape != self.data.shape:
                    grad_self = grad_self.reshape(self.data.shape)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = out.grad * self.data
                # Reduce to match other.data shape
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0, keepdims=False)
                if grad_other.shape != other.data.shape:
                    axes = [
                        i
                        for i in range(grad_other.ndim)
                        if i < len(other.data.shape) and other.data.shape[i] == 1
                    ]
                    if axes:
                        grad_other = grad_other.sum(axis=tuple(axes), keepdims=False)
                if grad_other.shape != other.data.shape:
                    grad_other = grad_other.reshape(other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        """Reverse multiplication (for int * Tensor)."""
        return self.__mul__(other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication (x @ W).

        For x (N, D) @ W (D, H) → out (N, H)
        """
        other = self._ensure_tensor(other)

        out = Tensor(
            self.data @ other.data,
            _children=(self, other),
            _name=f"{self._name}@{other._name}" if self._name else None,
        )

        def _backward():
            # ∂out/∂self = out.grad @ W^T
            # ∂out/∂other = x^T @ out.grad
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                # Ensure out.grad has correct shape (it should be batch_dim, out_features)
                grad = out.grad
                # If grad is 1D, reshape to (1, -1) for proper matmul
                if grad.ndim == 1:
                    grad = grad.reshape(1, -1)
                other.grad += self.data.T @ grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        """Negation."""
        return self * Tensor(-1.0, _name="neg_const")

    def __sub__(self, other: "Tensor") -> "Tensor":
        """Subtraction."""
        return self + (-other)

    def __rsub__(self, other) -> "Tensor":
        """Reverse subtraction."""
        return (-self) + other

    def __truediv__(self, other: "Tensor") -> "Tensor":
        """Element-wise division."""
        return self * other ** (-1)

    def __pow__(self, power: float) -> "Tensor":
        """Element-wise power."""
        out = Tensor(
            self.data**power,
            _children=(self,),
            _name=f"{self._name}^{power}" if self._name else None,
        )

        def _backward():
            # ∂out/∂self = power * self^(power-1)
            if self.requires_grad:
                self.grad += out.grad * power * (self.data ** (power - 1))

        out._backward = _backward
        return out

    def sum(self, axis: int | None = None) -> "Tensor":
        """
        Sum over the specified axis.

        Args:
            axis: Axis to sum over. If None, sum all elements.
        """
        out = Tensor(
            self.data.sum(axis=axis),
            _children=(self,),
            _name=f"sum_{self._name}" if self._name else None,
        )

        def _backward():
            if self.requires_grad:
                if axis is None:
                    # Scalar output: grad is broadcast to full shape
                    self.grad += np.full_like(self.data, out.grad)
                else:
                    # Sum over axis: expand dims and broadcast
                    expanded = np.expand_dims(out.grad, axis=axis)
                    # Tile to match original shape
                    repeat_counts = [
                        self.data.shape[i] if i == axis else 1
                        for i in range(self.data.ndim)
                    ]
                    self.grad += np.tile(expanded, repeat_counts)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        """Element-wise exponential."""
        out = Tensor(
            np.exp(self.data),
            _children=(self,),
            _name=f"exp({self._name})" if self._name else None,
        )

        def _backward():
            if self.requires_grad:
                # ∂out/∂self = exp(self) = out
                self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """Element-wise natural log."""
        out = Tensor(
            np.log(np.maximum(self.data, 1e-15)),  # Avoid log(0)
            _children=(self,),
            _name=f"log({self._name})" if self._name else None,
        )

        def _backward():
            if self.requires_grad:
                # ∂out/∂self = 1/self
                self.grad += out.grad / np.maximum(self.data, 1e-15)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        """Element-wise ReLU activation."""
        out = Tensor(
            np.maximum(0, self.data),
            _children=(self,),
            _name=f"relu({self._name})" if self._name else None,
        )

        def _backward():
            if self.requires_grad:
                # ∂out/∂self = 1 if self > 0 else 0
                mask = (self.data > 0).astype(np.float64)
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        """Element-wise tanh activation."""
        out = Tensor(
            np.tanh(self.data),
            _children=(self,),
            _name=f"tanh({self._name})" if self._name else None,
        )

        def _backward():
            if self.requires_grad:
                # ∂out/∂self = 1 - tanh²(self) = 1 - out²
                self.grad += out.grad * (1 - out.data**2)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        """Element-wise sigmoid activation."""
        out = Tensor(
            1 / (1 + np.exp(-np.clip(self.data, -500, 500))),  # Avoid overflow
            _children=(self,),
            _name=f"sigmoid({self._name})" if self._name else None,
        )

        def _backward():
            if self.requires_grad:
                # ∂out/∂self = out * (1 - out)
                self.grad += out.grad * out.data * (1 - out.data)

        out._backward = _backward
        return out

    # =====================================================================
    #  BACKPROPAGATION ENGINE
    # =====================================================================

    def backward(self, grad: np.ndarray | None = None) -> None:
        """
        Compute gradients using reverse-mode automatic differentiation.

        Args:
            grad: Optional external gradient to seed (for vector-valued outputs)
                If None, uses ones (for scalar loss) or zeros otherwise.
        """
        # Topological sort using DFS
        topo = []
        visited = set()

        def _build_topo(v: "Tensor") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        # Seed the gradient at the root
        if grad is not None:
            self.grad = grad.copy()
        else:
            # For scalar output: grad is 1
            if self.data.shape == () or self.data.size == 1:
                self.grad = np.array(1.0, dtype=np.float64)
            else:
                # For vector output: grad is all ones
                self.grad = np.ones_like(self.data, dtype=np.float64)

        # Reverse pass: call each node's _backward
        for v in reversed(topo):
            v._backward()

    # =====================================================================
    #  UTILITIES
    # =====================================================================

    def _ensure_tensor(self, other) -> "Tensor":
        """Convert a scalar or list to a Tensor if needed."""
        if not isinstance(other, Tensor):
            return Tensor(other, _name="const")
        return other

    def detach(self) -> "Tensor":
        """Return a new Tensor detached from the computation graph."""
        return Tensor(
            self.data.copy(), _name=f"detached_{self._name}" if self._name else None
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        return self.data.dtype

    def __getitem__(self, idx) -> "Tensor":
        """Indexing operation."""
        out = Tensor(
            self.data[idx],
            _children=(self,),
            _name=f"{self._name}[{idx}]" if self._name else None,
        )

        def _backward():
            if self.requires_grad:
                # Slicing backward: grad is placed back at the sliced positions
                self.grad[idx] += out.grad

        out._backward = _backward
        return out
