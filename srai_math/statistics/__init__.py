from .inference import (
    bootstrap_confidence_interval,
    normal_mean_confidence_interval,
    proportion_confidence_interval_wilson,
    required_sample_size_mean,
    required_sample_size_proportion,
    sample_mean,
    sample_variance,
    standard_error_mean,
    t_mean_confidence_interval,
)

__all__ = [
    "sample_mean",
    "sample_variance",
    "standard_error_mean",
    "normal_mean_confidence_interval",
    "t_mean_confidence_interval",
    "proportion_confidence_interval_wilson",
    "bootstrap_confidence_interval",
    "required_sample_size_mean",
    "required_sample_size_proportion",
]

from .testing import (
    cohen_d_independent,
    cohen_d_one_sample,
    cohen_d_paired,
    one_sample_t_test,
    paired_t_test,
    power_one_sample_z,
    proportion_z_test,
    two_sample_t_test,
    type_i_error_rate,
)

from .likelihood import (
    bernoulli_log_likelihood, bernoulli_mle, beta_bernoulli_posterior,
    beta_credible_interval, beta_posterior_mean, gamma_poisson_posterior,
    gamma_posterior_mean, map_beta_bernoulli, normal_log_likelihood,
    normal_mle, poisson_log_likelihood, poisson_mle,
)

# Compatibility API used by the canonical M1 notebook collection.  The
# package directory takes precedence over the historical statistics.py module,
# so these public names must be exported here as well.
import numpy as np
from scipy import stats


def _result(statistic, p_value):
    return {"statistic": float(statistic), "p_value": float(p_value)}


def independent_t_test(a, b, equal_variance=False, alternative="two-sided"):
    result = stats.ttest_ind(a, b, equal_var=equal_variance, alternative=alternative)
    return _result(result.statistic, result.pvalue)


def one_sample_proportion_z_test(successes, trials, null_proportion, alternative="two-sided"):
    statistic, p_value = proportion_z_test(successes, trials, null_proportion, alternative)
    return _result(statistic, p_value)


def cohens_d_one_sample(x, reference=0.0):
    return cohen_d_one_sample(x, reference)


def cohens_d_independent(a, b, pooled=True):
    return cohen_d_independent(a, b, pooled=pooled)


def hedges_g(a, b):
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return cohens_d_independent(a, b) * correction


def risk_difference(successes_treatment, total_treatment, successes_control, total_control):
    return successes_treatment / total_treatment - successes_control / total_control


def relative_risk(successes_treatment, total_treatment, successes_control, total_control):
    return (successes_treatment / total_treatment) / (successes_control / total_control)


def bonferroni_alpha(alpha, tests):
    return alpha / tests


def one_sample_mean_power(effect_size, sample_size, alpha=0.05):
    return power_one_sample_z(effect_size, sample_size, alpha)


def required_sample_size_one_sample_mean(effect_size, power=0.8, alpha=0.05):
    if effect_size <= 0 or not 0 < power < 1:
        raise ValueError("effect_size must be positive and power must lie in (0,1).")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return int(np.ceil(((z_alpha + z_power) / effect_size) ** 2))


# The notebook-facing API returns mappings, while the reusable testing module
# retains its tuple-returning scientific primitives.
_one_sample_t_test = one_sample_t_test
_paired_t_test = paired_t_test


def one_sample_t_test(x, null_mean=0.0, alternative="two-sided"):
    statistic, p_value = _one_sample_t_test(x, null_mean, alternative)
    return _result(statistic, p_value)


def paired_t_test(before, after, alternative="two-sided"):
    statistic, p_value = _paired_t_test(before, after, alternative)
    return _result(statistic, p_value)
