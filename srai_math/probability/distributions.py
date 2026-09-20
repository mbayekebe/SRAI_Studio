"""Random-variable and common-distribution utilities."""

from __future__ import annotations

from math import comb, exp, factorial, pi, sqrt
import numpy as np
from numpy.typing import ArrayLike, NDArray


def bernoulli_pmf(x: int, p: float) -> float:
    if x not in (0, 1):
        return 0.0
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1].")
    return float(p if x == 1 else 1 - p)


def binomial_pmf(k: int, n: int, p: float) -> float:
    if n < 0 or not isinstance(n, int):
        raise ValueError("n must be a non-negative integer.")
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1].")
    if k < 0 or k > n or not isinstance(k, int):
        return 0.0
    return float(comb(n, k) * (p**k) * ((1 - p) ** (n - k)))


def poisson_pmf(k: int, rate: float) -> float:
    if rate <= 0:
        raise ValueError("rate must be positive.")
    if k < 0 or not isinstance(k, int):
        return 0.0
    return float(exp(-rate) * (rate**k) / factorial(k))


def uniform_pdf(x: ArrayLike, a: float, b: float) -> NDArray[np.float64]:
    if b <= a:
        raise ValueError("Require b > a.")
    values = np.asarray(x, dtype=float)
    return np.where((values >= a) & (values <= b), 1.0 / (b - a), 0.0)


def normal_pdf(x: ArrayLike, mean: float = 0.0, std: float = 1.0) -> NDArray[np.float64]:
    if std <= 0:
        raise ValueError("std must be positive.")
    values = np.asarray(x, dtype=float)
    coefficient = 1.0 / (std * sqrt(2.0 * pi))
    exponent = -0.5 * ((values - mean) / std) ** 2
    return coefficient * np.exp(exponent)


def exponential_pdf(x: ArrayLike, rate: float) -> NDArray[np.float64]:
    if rate <= 0:
        raise ValueError("rate must be positive.")
    values = np.asarray(x, dtype=float)
    return np.where(values >= 0, rate * np.exp(-rate * values), 0.0)


def empirical_cdf(samples: ArrayLike, points: ArrayLike) -> NDArray[np.float64]:
    data = np.asarray(samples, dtype=float)
    query = np.asarray(points, dtype=float)
    if data.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    return np.asarray([np.mean(data <= point) for point in query], dtype=float)


def sample_distribution(
    name: str,
    size: int,
    seed: int = 42,
    **parameters,
) -> NDArray[np.float64]:
    if size <= 0:
        raise ValueError("size must be positive.")
    rng = np.random.default_rng(seed)
    if name == "bernoulli":
        return rng.binomial(1, parameters["p"], size=size).astype(float)
    if name == "binomial":
        return rng.binomial(parameters["n"], parameters["p"], size=size).astype(float)
    if name == "poisson":
        return rng.poisson(parameters["rate"], size=size).astype(float)
    if name == "uniform":
        return rng.uniform(parameters["a"], parameters["b"], size=size)
    if name == "normal":
        return rng.normal(parameters.get("mean", 0.0), parameters.get("std", 1.0), size=size)
    if name == "exponential":
        return rng.exponential(1.0 / parameters["rate"], size=size)
    raise ValueError("Unsupported distribution name.")


def sample_moments(samples: ArrayLike) -> dict[str, float]:
    data = np.asarray(samples, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array.")
    centered = data - np.mean(data)
    variance = np.mean(centered**2)
    std = np.sqrt(variance)
    skewness = 0.0 if np.isclose(std, 0.0) else float(np.mean(centered**3) / std**3)
    return {
        "mean": float(np.mean(data)),
        "variance": float(variance),
        "standard_deviation": float(std),
        "skewness": skewness,
    }
