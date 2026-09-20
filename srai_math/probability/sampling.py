"""Sampling distributions and Central Limit Theorem utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sample_means(
    sampler: Callable[[tuple[int, int]], NDArray[np.float64]],
    sample_size: int,
    repetitions: int,
) -> NDArray[np.float64]:
    """Generate a sampling distribution of sample means."""
    if sample_size <= 0 or repetitions <= 0:
        raise ValueError("sample_size and repetitions must be positive.")
    draws = np.asarray(sampler((repetitions, sample_size)), dtype=float)
    if draws.shape != (repetitions, sample_size):
        raise ValueError("sampler must return shape (repetitions, sample_size).")
    return draws.mean(axis=1)


def sampling_standard_error(population_std: float, sample_size: int) -> float:
    """Return sigma / sqrt(n)."""
    if population_std < 0:
        raise ValueError("population_std must be non-negative.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    return float(population_std / np.sqrt(sample_size))


def standardize_sample_means(
    means: ArrayLike,
    population_mean: float,
    population_std: float,
    sample_size: int,
) -> NDArray[np.float64]:
    """Standardize sample means using the theoretical standard error."""
    values = np.asarray(means, dtype=float)
    se = sampling_standard_error(population_std, sample_size)
    if np.isclose(se, 0.0):
        raise ValueError("Standard error must be positive.")
    return (values - population_mean) / se


def bootstrap_statistic(
    data: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float],
    repetitions: int = 1000,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate a nonparametric bootstrap distribution."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("data must be a non-empty one-dimensional array.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    rng = np.random.default_rng(seed)
    result = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        sample = rng.choice(values, size=values.size, replace=True)
        result[i] = float(statistic(sample))
    return result


def bootstrap_standard_error(
    bootstrap_values: ArrayLike,
    ddof: int = 1,
) -> float:
    """Return the standard deviation of bootstrap replicates."""
    values = np.asarray(bootstrap_values, dtype=float)
    if values.ndim != 1 or values.size <= ddof:
        raise ValueError("Not enough bootstrap values.")
    return float(np.std(values, ddof=ddof))


def finite_population_correction(
    population_size: int,
    sample_size: int,
) -> float:
    """Return sqrt((N-n)/(N-1))."""
    if population_size <= 1:
        raise ValueError("population_size must exceed one.")
    if not 0 < sample_size <= population_size:
        raise ValueError("sample_size must satisfy 0 < n <= N.")
    return float(np.sqrt((population_size - sample_size) / (population_size - 1)))
