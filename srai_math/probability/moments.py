"""Expectation, variance, covariance, and concentration utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def discrete_expectation(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Return E[X] for a finite discrete distribution."""
    x = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if x.ndim != 1 or p.ndim != 1 or x.shape != p.shape:
        raise ValueError("values and probabilities must be equally shaped vectors.")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("probabilities must be non-negative and sum to one.")
    return float(x @ p)


def discrete_variance(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Return Var(X) for a finite discrete distribution."""
    x = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    mean = discrete_expectation(x, p)
    return float(((x - mean) ** 2) @ p)


def covariance(
    x: ArrayLike,
    y: ArrayLike,
    ddof: int = 1,
) -> float:
    """Return sample or population covariance."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("x and y must be equally shaped vectors.")
    if a.size - ddof <= 0:
        raise ValueError("Not enough observations for requested ddof.")
    return float(np.sum((a - a.mean()) * (b - b.mean())) / (a.size - ddof))


def correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Return Pearson correlation."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("x and y must be equally shaped vectors.")
    sa = np.std(a, ddof=1)
    sb = np.std(b, ddof=1)
    if np.isclose(sa, 0.0) or np.isclose(sb, 0.0):
        raise ValueError("Correlation is undefined for a constant variable.")
    return float(covariance(a, b, ddof=1) / (sa * sb))


def covariance_matrix(x: ArrayLike, rowvar: bool = False) -> NDArray[np.float64]:
    """Return a covariance matrix."""
    data = np.asarray(x, dtype=float)
    if data.ndim != 2:
        raise ValueError("x must be a matrix.")
    return np.cov(data, rowvar=rowvar, ddof=1)


def chebyshev_bound(variance: float, deviation: float) -> float:
    """Return Chebyshev upper bound P(|X-mu| >= deviation)."""
    if variance < 0:
        raise ValueError("variance must be non-negative.")
    if deviation <= 0:
        raise ValueError("deviation must be positive.")
    return float(min(1.0, variance / (deviation**2)))


def markov_bound(expectation: float, threshold: float) -> float:
    """Return Markov upper bound P(X >= threshold) for X >= 0."""
    if expectation < 0:
        raise ValueError("expectation must be non-negative.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    return float(min(1.0, expectation / threshold))


def hoeffding_bound(
    epsilon: float,
    n: int,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Return two-sided Hoeffding bound for bounded independent variables."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if upper <= lower:
        raise ValueError("upper must exceed lower.")
    width = upper - lower
    return float(min(1.0, 2.0 * np.exp(-2.0 * n * epsilon**2 / width**2)))


def running_mean(x: ArrayLike) -> NDArray[np.float64]:
    """Return cumulative sample means."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty vector.")
    return np.cumsum(values) / np.arange(1, values.size + 1)


def running_variance(x: ArrayLike) -> NDArray[np.float64]:
    """Return cumulative population variances."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty vector.")
    result = np.zeros(values.size, dtype=float)
    for i in range(1, values.size + 1):
        result[i - 1] = np.var(values[:i], ddof=0)
    return result
