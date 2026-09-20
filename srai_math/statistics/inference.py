"""Point estimation and confidence interval utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def sample_mean(x: ArrayLike) -> float:
    """Return the arithmetic mean."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array.")
    return float(np.mean(values))


def sample_variance(x: ArrayLike, ddof: int = 1) -> float:
    """Return sample variance."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size <= ddof:
        raise ValueError("Not enough observations.")
    return float(np.var(values, ddof=ddof))


def standard_error_mean(x: ArrayLike) -> float:
    """Return s/sqrt(n)."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two observations are required.")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def normal_mean_confidence_interval(
    mean: float,
    population_std: float,
    sample_size: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval for a mean with known population standard deviation."""
    if population_std < 0:
        raise ValueError("population_std must be non-negative.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1).")
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    margin = z * population_std / np.sqrt(sample_size)
    return float(mean - margin), float(mean + margin)


def t_mean_confidence_interval(
    x: ArrayLike,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Student-t confidence interval for a population mean."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two observations are required.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1).")
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(values.size)
    alpha = 1.0 - confidence
    critical = stats.t.ppf(1.0 - alpha / 2.0, df=values.size - 1)
    margin = critical * se
    return float(mean - margin), float(mean + margin)


def proportion_confidence_interval_wilson(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson confidence interval for a binomial proportion."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= trials and trials > 0.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1).")
    p = successes / trials
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2.0 * trials)) / denominator
    margin = (
        z
        * np.sqrt(p * (1.0 - p) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return float(center - margin), float(center + margin)


def bootstrap_confidence_interval(
    data: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float],
    confidence: float = 0.95,
    repetitions: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("data must be a non-empty vector.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1).")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[i] = float(statistic(sample))
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def required_sample_size_mean(
    population_std: float,
    margin_error: float,
    confidence: float = 0.95,
) -> int:
    """Required sample size for a mean with known population standard deviation."""
    if population_std <= 0 or margin_error <= 0:
        raise ValueError("population_std and margin_error must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1).")
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return int(np.ceil((z * population_std / margin_error) ** 2))


def required_sample_size_proportion(
    margin_error: float,
    confidence: float = 0.95,
    anticipated_proportion: float = 0.5,
) -> int:
    """Required sample size for estimating a proportion."""
    if margin_error <= 0:
        raise ValueError("margin_error must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1).")
    if not 0 <= anticipated_proportion <= 1:
        raise ValueError("anticipated_proportion must lie in [0,1].")
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = anticipated_proportion
    return int(np.ceil(z**2 * p * (1 - p) / margin_error**2))
