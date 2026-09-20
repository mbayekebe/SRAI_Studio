"""Conditional probability and Bayesian updating utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Hashable

import numpy as np


def bayes_binary(
    prior: float,
    sensitivity: float,
    specificity: float,
    positive: bool = True,
) -> float:
    """Return posterior P(condition | test result) for a binary test."""
    for name, value in {
        "prior": prior,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1].")

    if positive:
        numerator = sensitivity * prior
        denominator = numerator + (1 - specificity) * (1 - prior)
    else:
        numerator = (1 - sensitivity) * prior
        denominator = numerator + specificity * (1 - prior)

    if np.isclose(denominator, 0.0):
        raise ZeroDivisionError("Evidence has zero probability.")
    return float(numerator / denominator)


def total_probability(
    priors: Iterable[float],
    likelihoods: Iterable[float],
) -> float:
    """Return the total probability of evidence."""
    p = np.asarray(list(priors), dtype=float)
    l = np.asarray(list(likelihoods), dtype=float)
    if p.ndim != 1 or l.ndim != 1 or p.shape != l.shape:
        raise ValueError("priors and likelihoods must have equal one-dimensional shape.")
    if np.any(p < 0) or np.any(l < 0):
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(p.sum(), 1.0):
        raise ValueError("priors must sum to one.")
    return float(p @ l)


def bayes_discrete(
    priors: Mapping[Hashable, float],
    likelihoods: Mapping[Hashable, float],
) -> dict[Hashable, float]:
    """Update a discrete hypothesis distribution from likelihoods."""
    if set(priors) != set(likelihoods):
        raise ValueError("priors and likelihoods must have identical hypotheses.")
    prior_values = np.asarray(list(priors.values()), dtype=float)
    likelihood_values = np.asarray(list(likelihoods.values()), dtype=float)
    if np.any(prior_values < 0) or np.any(likelihood_values < 0):
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(prior_values.sum(), 1.0):
        raise ValueError("priors must sum to one.")

    unnormalized = prior_values * likelihood_values
    evidence = float(unnormalized.sum())
    if np.isclose(evidence, 0.0):
        raise ZeroDivisionError("Evidence has zero probability.")

    posterior = unnormalized / evidence
    return {
        hypothesis: float(value)
        for hypothesis, value in zip(priors.keys(), posterior)
    }


def sequential_bayes(
    prior: Mapping[Hashable, float],
    evidence_likelihoods: Iterable[Mapping[Hashable, float]],
) -> list[dict[Hashable, float]]:
    """Apply a sequence of discrete Bayesian updates."""
    current = dict(prior)
    history = [current.copy()]
    for likelihood in evidence_likelihoods:
        current = bayes_discrete(current, likelihood)
        history.append(current.copy())
    return history


def posterior_odds(
    prior_odds: float,
    likelihood_ratio: float,
) -> float:
    """Return posterior odds = prior odds × likelihood ratio."""
    if prior_odds < 0 or likelihood_ratio < 0:
        raise ValueError("Odds and likelihood ratio must be non-negative.")
    return float(prior_odds * likelihood_ratio)


def probability_to_odds(probability: float) -> float:
    """Convert probability to odds."""
    if not 0 <= probability < 1:
        raise ValueError("probability must lie in [0, 1).")
    return float(probability / (1 - probability))


def odds_to_probability(odds: float) -> float:
    """Convert odds to probability."""
    if odds < 0:
        raise ValueError("odds must be non-negative.")
    return float(odds / (1 + odds))
