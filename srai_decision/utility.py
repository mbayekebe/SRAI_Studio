"""Utility and value functions."""
from __future__ import annotations
import numpy as np

def linear_utility(x,minimum,maximum):
    x=np.asarray(x,float)
    return np.clip((x-minimum)/(maximum-minimum+1e-12),0,1)

def decreasing_utility(x,minimum,maximum):
    return 1-linear_utility(x,minimum,maximum)

def exponential_utility(x,risk_aversion=.1):
    x=np.asarray(x,float)
    return 1-np.exp(-risk_aversion*x)

def expected_utility(outcomes,probabilities,utility_function=lambda x:x):
    outcomes=np.asarray(outcomes,float)
    probabilities=np.asarray(probabilities,float)
    utilities=np.asarray(utility_function(outcomes),float)
    return float(np.sum(probabilities*utilities))

def certainty_equivalent(expected_utility_value,risk_aversion=.1):
    if risk_aversion<=0:
        return float(expected_utility_value)
    return float(-np.log(max(1-expected_utility_value,1e-12))/risk_aversion)

def value_of_information(with_information,without_information):
    return float(with_information-without_information)
