"""Small forward-mode automatic differentiation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos as math_cos
from math import exp as math_exp
from math import log as math_log
from math import sin as math_sin
from typing import Union

import numpy as np

Number = Union[int, float]


@dataclass(frozen=True)
class Dual:
    """Dual number x + epsilon*x' for forward-mode automatic differentiation."""

    value: float
    derivative: float = 0.0

    def __add__(self, other: Number | "Dual") -> "Dual":
        other = ensure_dual(other)
        return Dual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __sub__(self, other: Number | "Dual") -> "Dual":
        other = ensure_dual(other)
        return Dual(self.value - other.value, self.derivative - other.derivative)

    def __rsub__(self, other: Number | "Dual") -> "Dual":
        other = ensure_dual(other)
        return other - self

    def __mul__(self, other: Number | "Dual") -> "Dual":
        other = ensure_dual(other)
        return Dual(
            self.value * other.value,
            self.derivative * other.value + self.value * other.derivative,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Number | "Dual") -> "Dual":
        other = ensure_dual(other)
        if np.isclose(other.value, 0.0):
            raise ZeroDivisionError("Division by zero dual number.")
        return Dual(
            self.value / other.value,
            (
                self.derivative * other.value
                - self.value * other.derivative
            )
            / (other.value**2),
        )

    def __rtruediv__(self, other: Number | "Dual") -> "Dual":
        other = ensure_dual(other)
        return other / self

    def __pow__(self, power: Number) -> "Dual":
        if not isinstance(power, (int, float)):
            raise TypeError("Only constant scalar powers are supported.")
        if self.value == 0 and power < 1:
            raise ValueError("Undefined derivative for this power at zero.")
        return Dual(
            self.value**power,
            power * (self.value ** (power - 1)) * self.derivative,
        )

    def __neg__(self) -> "Dual":
        return Dual(-self.value, -self.derivative)


def ensure_dual(x: Number | Dual) -> Dual:
    """Convert a scalar to a constant dual number."""
    return x if isinstance(x, Dual) else Dual(float(x), 0.0)


def dual_sin(x: Number | Dual) -> Dual:
    x = ensure_dual(x)
    return Dual(math_sin(x.value), math_cos(x.value) * x.derivative)


def dual_cos(x: Number | Dual) -> Dual:
    x = ensure_dual(x)
    return Dual(math_cos(x.value), -math_sin(x.value) * x.derivative)


def dual_exp(x: Number | Dual) -> Dual:
    x = ensure_dual(x)
    value = math_exp(x.value)
    return Dual(value, value * x.derivative)


def dual_log(x: Number | Dual) -> Dual:
    x = ensure_dual(x)
    if x.value <= 0:
        raise ValueError("log requires a positive value.")
    return Dual(math_log(x.value), x.derivative / x.value)


def autodiff_derivative(function, x: float) -> tuple[float, float]:
    """Return function value and derivative using a seeded dual number."""
    result = function(Dual(float(x), 1.0))
    if not isinstance(result, Dual):
        raise TypeError("function must return a Dual when supplied a Dual.")
    return result.value, result.derivative


def autodiff_gradient(function, x) -> tuple[float, np.ndarray]:
    """Compute a gradient by one forward-mode pass per input dimension."""
    point = np.asarray(x, dtype=float)
    if point.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    gradient = np.zeros_like(point)
    value = None

    for i in range(point.size):
        dual_inputs = [
            Dual(float(point[j]), 1.0 if i == j else 0.0)
            for j in range(point.size)
        ]
        result = function(dual_inputs)
        if not isinstance(result, Dual):
            raise TypeError("function must return a Dual.")
        if value is None:
            value = result.value
        gradient[i] = result.derivative

    return float(value), gradient


def relative_gradient_error(analytic, numerical, epsilon: float = 1e-12) -> float:
    """Return a symmetric relative error between two gradient vectors."""
    a = np.asarray(analytic, dtype=float)
    n = np.asarray(numerical, dtype=float)
    if a.shape != n.shape:
        raise ValueError("Gradient vectors must have identical shapes.")
    denominator = max(np.linalg.norm(a) + np.linalg.norm(n), epsilon)
    return float(np.linalg.norm(a - n) / denominator)


def gradient_check(analytic, numerical, tolerance: float = 1e-6) -> bool:
    """Return True when relative gradient error is within tolerance."""
    return relative_gradient_error(analytic, numerical) <= tolerance
