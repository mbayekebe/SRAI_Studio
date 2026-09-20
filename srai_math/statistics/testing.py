"""Hypothesis testing, power, and effect-size utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def one_sample_t_test(
    x: ArrayLike,
    null_mean: float,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Return one-sample t statistic and p-value."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("x must contain at least two observations.")
    result = stats.ttest_1samp(values, popmean=null_mean, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def two_sample_t_test(
    x: ArrayLike,
    y: ArrayLike,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Return independent two-sample t statistic and p-value."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.size < 2 or b.size < 2:
        raise ValueError("x and y must each contain at least two observations.")
    result = stats.ttest_ind(
        a,
        b,
        equal_var=equal_var,
        alternative=alternative,
    )
    return float(result.statistic), float(result.pvalue)


def paired_t_test(
    before: ArrayLike,
    after: ArrayLike,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Return paired t statistic and p-value."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("before and after must be equally shaped vectors.")
    result = stats.ttest_rel(a, b, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def cohen_d_one_sample(x: ArrayLike, null_mean: float) -> float:
    """Return one-sample Cohen's d."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("x must contain at least two observations.")
    s = np.std(values, ddof=1)
    if np.isclose(s, 0.0):
        raise ValueError("Effect size undefined for zero variance.")
    return float((np.mean(values) - null_mean) / s)


def cohen_d_independent(
    x: ArrayLike,
    y: ArrayLike,
    pooled: bool = True,
) -> float:
    """Return Cohen's d for two independent groups."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.size < 2 or b.size < 2:
        raise ValueError("x and y must each contain at least two observations.")
    mean_diff = np.mean(a) - np.mean(b)
    if pooled:
        pooled_var = (
            (a.size - 1) * np.var(a, ddof=1)
            + (b.size - 1) * np.var(b, ddof=1)
        ) / (a.size + b.size - 2)
        denominator = np.sqrt(pooled_var)
    else:
        denominator = np.sqrt(
            (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
        )
    if np.isclose(denominator, 0.0):
        raise ValueError("Effect size undefined for zero variance.")
    return float(mean_diff / denominator)


def cohen_d_paired(before: ArrayLike, after: ArrayLike) -> float:
    """Return standardized mean difference for paired observations."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("before and after must be equally shaped vectors.")
    differences = b - a
    sd = np.std(differences, ddof=1)
    if np.isclose(sd, 0.0):
        mean_difference = float(np.mean(differences))
        if np.isclose(mean_difference, 0.0):
            return 0.0
        return float(np.copysign(np.inf, mean_difference))
    return float(np.mean(differences) / sd)


def proportion_z_test(
    successes: int,
    trials: int,
    null_proportion: float,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """One-sample z test for a proportion."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= trials and trials > 0.")
    if not 0 < null_proportion < 1:
        raise ValueError("null_proportion must lie in (0,1).")
    p_hat = successes / trials
    se = np.sqrt(null_proportion * (1 - null_proportion) / trials)
    z = (p_hat - null_proportion) / se
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p = stats.norm.sf(z)
    elif alternative == "less":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("Unsupported alternative.")
    return float(z), float(p)


def type_i_error_rate(
    test_function,
    sampler,
    repetitions: int = 5000,
) -> float:
    """Estimate rejection rate under a null data-generating process."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    rejections = 0
    for _ in range(repetitions):
        sample = sampler()
        _, p_value = test_function(sample)
        rejections += p_value < 0.05
    return float(rejections / repetitions)


def power_one_sample_z(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Approximate power for a one-sample z test using standardized effect size."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1).")
    noncentral = effect_size * np.sqrt(sample_size)
    if alternative == "two-sided":
        critical = stats.norm.ppf(1 - alpha / 2)
        return float(
            stats.norm.cdf(-critical - noncentral)
            + stats.norm.sf(critical - noncentral)
        )
    if alternative == "greater":
        critical = stats.norm.ppf(1 - alpha)
        return float(stats.norm.sf(critical - noncentral))
    if alternative == "less":
        critical = stats.norm.ppf(alpha)
        return float(stats.norm.cdf(critical - noncentral))
    raise ValueError("Unsupported alternative.")
