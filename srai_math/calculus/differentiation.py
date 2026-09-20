"""Numerical differentiation utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def derivative(
    function: Callable[[float], float],
    x: float,
    h: float = 1e-5,
    method: str = "central",
) -> float:
    """Estimate a scalar derivative with finite differences."""
    if h <= 0:
        raise ValueError("h must be positive.")
    if method == "forward":
        return float((function(x + h) - function(x)) / h)
    if method == "backward":
        return float((function(x) - function(x - h)) / h)
    if method == "central":
        return float((function(x + h) - function(x - h)) / (2.0 * h))
    raise ValueError("method must be 'forward', 'backward', or 'central'.")


def second_derivative(
    function: Callable[[float], float],
    x: float,
    h: float = 1e-4,
) -> float:
    """Estimate a scalar second derivative."""
    if h <= 0:
        raise ValueError("h must be positive.")
    return float((function(x + h) - 2.0 * function(x) + function(x - h)) / (h**2))


def gradient(
    function: Callable[[NDArray[np.float64]], float],
    x: ArrayLike,
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Estimate the gradient of a scalar-valued multivariate function."""
    point = np.asarray(x, dtype=float)
    if point.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if h <= 0:
        raise ValueError("h must be positive.")
    result = np.zeros_like(point)
    for i in range(point.size):
        step = np.zeros_like(point)
        step[i] = h
        result[i] = (function(point + step) - function(point - step)) / (2.0 * h)
    return result


def jacobian(
    function: Callable[[NDArray[np.float64]], ArrayLike],
    x: ArrayLike,
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Estimate the Jacobian of a vector-valued function."""
    point = np.asarray(x, dtype=float)
    if point.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    base = np.asarray(function(point), dtype=float)
    if base.ndim != 1:
        raise ValueError("function output must be one-dimensional.")
    J = np.zeros((base.size, point.size), dtype=float)
    for i in range(point.size):
        step = np.zeros_like(point)
        step[i] = h
        J[:, i] = (
            np.asarray(function(point + step), dtype=float)
            - np.asarray(function(point - step), dtype=float)
        ) / (2.0 * h)
    return J


def hessian(
    function: Callable[[NDArray[np.float64]], float],
    x: ArrayLike,
    h: float = 1e-4,
) -> NDArray[np.float64]:
    """Estimate the Hessian matrix of a scalar-valued function."""
    point = np.asarray(x, dtype=float)
    if point.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    n = point.size
    H = np.zeros((n, n), dtype=float)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        H[i, i] = (
            function(point + ei)
            - 2.0 * function(point)
            + function(point - ei)
        ) / (h**2)

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h
            value = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4.0 * h**2)
            H[i, j] = value
            H[j, i] = value
    return H


def directional_derivative(
    function: Callable[[NDArray[np.float64]], float],
    x: ArrayLike,
    direction: ArrayLike,
    h: float = 1e-5,
) -> float:
    """Estimate a directional derivative along a normalized direction."""
    point = np.asarray(x, dtype=float)
    d = np.asarray(direction, dtype=float)
    if point.ndim != 1 or d.ndim != 1 or point.shape != d.shape:
        raise ValueError("x and direction must be equally shaped vectors.")
    norm = np.linalg.norm(d)
    if np.isclose(norm, 0.0):
        raise ValueError("direction must be nonzero.")
    unit = d / norm
    return float((function(point + h * unit) - function(point - h * unit)) / (2.0 * h))


def taylor_first_order(
    function_value: float,
    gradient_value: ArrayLike,
    x: ArrayLike,
    x0: ArrayLike,
) -> float:
    """Evaluate a first-order Taylor approximation."""
    g = np.asarray(gradient_value, dtype=float)
    point = np.asarray(x, dtype=float)
    center = np.asarray(x0, dtype=float)
    if g.shape != point.shape or point.shape != center.shape:
        raise ValueError("gradient, x, and x0 must have identical shapes.")
    return float(function_value + g @ (point - center))
