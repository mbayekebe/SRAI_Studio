"""Numerical utilities for limits and continuity."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray


def approach_sequence(
    point: float,
    scales: Iterable[float] | None = None,
    side: str = "both",
) -> NDArray[np.float64]:
    """Generate values approaching a point from the left, right, or both."""
    if scales is None:
        scales = [10.0 ** (-k) for k in range(1, 8)]
    steps = np.asarray(list(scales), dtype=float)
    if np.any(steps <= 0):
        raise ValueError("All approach scales must be positive.")
    if side == "left":
        return point - steps
    if side == "right":
        return point + steps
    if side == "both":
        return np.concatenate([point - steps[::-1], point + steps])
    raise ValueError("side must be 'left', 'right', or 'both'.")


def numerical_limit(
    function: Callable[[float], float],
    point: float,
    side: str = "both",
    scales: Iterable[float] | None = None,
) -> float:
    """Estimate a finite limit from nearby evaluations."""
    xs = approach_sequence(point, scales=scales, side=side)
    values = np.asarray([function(float(x)) for x in xs], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Function evaluations must be finite near the point.")

    if side == "both":
        half = len(values) // 2
        left = values[:half]
        right = values[half:]
        estimate = 0.5 * (left[0] + right[-1])
    else:
        estimate = values[-1]
    return float(estimate)


def one_sided_limits(
    function: Callable[[float], float],
    point: float,
    scales: Iterable[float] | None = None,
) -> tuple[float, float]:
    """Estimate left and right limits."""
    left = numerical_limit(function, point, side="left", scales=scales)
    right = numerical_limit(function, point, side="right", scales=scales)
    return left, right


def is_continuous_at(
    function: Callable[[float], float],
    point: float,
    tolerance: float = 1e-5,
    scales: Iterable[float] | None = None,
) -> bool:
    """Numerically test continuity at a point."""
    left, right = one_sided_limits(function, point, scales=scales)
    value = float(function(point))
    return bool(
        np.isfinite(value)
        and abs(left - right) <= tolerance
        and abs(value - left) <= tolerance
        and abs(value - right) <= tolerance
    )


def epsilon_delta_check(
    function: Callable[[float], float],
    point: float,
    limit_value: float,
    epsilon: float,
    delta: float,
    samples: int = 1000,
) -> bool:
    """Sample the punctured delta-neighborhood to test an epsilon condition."""
    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be positive.")
    left = np.linspace(point - delta, point, samples // 2, endpoint=False)
    right = np.linspace(point, point + delta, samples // 2 + 1)[1:]
    xs = np.concatenate([left, right])
    values = np.asarray([function(float(x)) for x in xs], dtype=float)
    return bool(np.all(np.abs(values - limit_value) < epsilon))


def removable_extension(
    function: Callable[[float], float],
    point: float,
    limit_value: float,
) -> Callable[[float], float]:
    """Return a function with a removable discontinuity filled in."""
    def extended(x: float) -> float:
        if np.isclose(x, point):
            return float(limit_value)
        return float(function(x))
    return extended
