"""Numerical integration and accumulation utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def left_riemann(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64] | float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Approximate an integral with left-endpoint rectangles."""
    if n <= 0:
        raise ValueError("n must be positive.")
    x = np.linspace(a, b, n + 1)
    width = (b - a) / n
    return float(width * np.sum(function(x[:-1])))


def right_riemann(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64] | float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Approximate an integral with right-endpoint rectangles."""
    if n <= 0:
        raise ValueError("n must be positive.")
    x = np.linspace(a, b, n + 1)
    width = (b - a) / n
    return float(width * np.sum(function(x[1:])))


def midpoint_rule(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64] | float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Approximate an integral with the midpoint rule."""
    if n <= 0:
        raise ValueError("n must be positive.")
    edges = np.linspace(a, b, n + 1)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    width = (b - a) / n
    return float(width * np.sum(function(midpoints)))


def trapezoidal_rule(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64] | float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Approximate an integral with the composite trapezoidal rule."""
    if n <= 0:
        raise ValueError("n must be positive.")
    x = np.linspace(a, b, n + 1)
    y = np.asarray(function(x), dtype=float)
    width = (b - a) / n
    return float(width * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def simpson_rule(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64] | float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Approximate an integral with composite Simpson's rule."""
    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive even integer.")
    x = np.linspace(a, b, n + 1)
    y = np.asarray(function(x), dtype=float)
    width = (b - a) / n
    return float(
        width / 3.0
        * (
            y[0]
            + y[-1]
            + 4.0 * np.sum(y[1:-1:2])
            + 2.0 * np.sum(y[2:-2:2])
        )
    )


def cumulative_trapezoid(
    y: ArrayLike,
    x: ArrayLike,
) -> NDArray[np.float64]:
    """Return cumulative trapezoidal integrals, beginning at zero."""
    values = np.asarray(y, dtype=float)
    grid = np.asarray(x, dtype=float)
    if values.ndim != 1 or grid.ndim != 1 or values.shape != grid.shape:
        raise ValueError("x and y must be equally shaped one-dimensional arrays.")
    if values.size < 2:
        raise ValueError("At least two points are required.")
    widths = np.diff(grid)
    areas = 0.5 * widths * (values[:-1] + values[1:])
    return np.concatenate([[0.0], np.cumsum(areas)])


def monte_carlo_integral(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64] | float],
    a: float,
    b: float,
    samples: int = 100_000,
    seed: int = 42,
) -> float:
    """Estimate a one-dimensional integral by uniform Monte Carlo sampling."""
    if samples <= 0:
        raise ValueError("samples must be positive.")
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=samples)
    return float((b - a) * np.mean(function(x)))
