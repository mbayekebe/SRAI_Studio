"""Finite probability spaces and simulation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Hashable

import numpy as np


@dataclass(frozen=True)
class FiniteProbabilitySpace:
    """A finite sample space with explicit outcome probabilities."""

    probabilities: Mapping[Hashable, float]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("Probability space cannot be empty.")
        values = np.asarray(list(self.probabilities.values()), dtype=float)
        if np.any(values < 0):
            raise ValueError("Probabilities must be non-negative.")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("Probabilities must sum to one.")

    @property
    def outcomes(self) -> set[Hashable]:
        return set(self.probabilities.keys())

    def probability(self, event: Iterable[Hashable]) -> float:
        """Return P(event)."""
        event_set = set(event)
        if not event_set <= self.outcomes:
            raise ValueError("Event contains outcomes outside the sample space.")
        return float(sum(self.probabilities[o] for o in event_set))

    def conditional_probability(
        self,
        event: Iterable[Hashable],
        given: Iterable[Hashable],
    ) -> float:
        """Return P(event | given)."""
        event_set = set(event)
        given_set = set(given)
        p_given = self.probability(given_set)
        if np.isclose(p_given, 0.0):
            raise ZeroDivisionError("Conditioning event has probability zero.")
        return self.probability(event_set & given_set) / p_given

    def independent(
        self,
        event_a: Iterable[Hashable],
        event_b: Iterable[Hashable],
        tolerance: float = 1e-12,
    ) -> bool:
        """Check whether two events are independent."""
        A = set(event_a)
        B = set(event_b)
        lhs = self.probability(A & B)
        rhs = self.probability(A) * self.probability(B)
        return bool(abs(lhs - rhs) <= tolerance)


def empirical_probability(
    observations: Iterable[Hashable],
    event: Iterable[Hashable],
) -> float:
    """Estimate event probability from observed outcomes."""
    data = list(observations)
    if not data:
        raise ValueError("observations cannot be empty.")
    event_set = set(event)
    return float(sum(item in event_set for item in data) / len(data))


def simulate_bernoulli(
    probability: float,
    size: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate Bernoulli outcomes."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1].")
    if size <= 0:
        raise ValueError("size must be positive.")
    rng = np.random.default_rng(seed)
    return rng.binomial(1, probability, size=size)


def simulate_categorical(
    outcomes,
    probabilities,
    size: int,
    seed: int = 42,
):
    """Simulate categorical outcomes."""
    values = np.asarray(list(outcomes), dtype=object)
    probs = np.asarray(list(probabilities), dtype=float)
    if values.size == 0 or values.size != probs.size:
        raise ValueError("outcomes and probabilities must have equal nonzero length.")
    if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("probabilities must be non-negative and sum to one.")
    if size <= 0:
        raise ValueError("size must be positive.")
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=size, p=probs)


def law_of_large_numbers_path(
    probability: float,
    size: int,
    seed: int = 42,
) -> np.ndarray:
    """Return cumulative Bernoulli means."""
    samples = simulate_bernoulli(probability, size=size, seed=seed)
    return np.cumsum(samples) / np.arange(1, size + 1)


def monte_carlo_expectation(
    function,
    sampler,
    samples: int,
) -> float:
    """Estimate E[f(X)] from sampler(samples)."""
    if samples <= 0:
        raise ValueError("samples must be positive.")
    draws = sampler(samples)
    values = np.asarray(function(draws), dtype=float)
    return float(np.mean(values))
