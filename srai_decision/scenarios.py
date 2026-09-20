"""Scenario-based decision analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd

def scenario_payoff(alternative_values,scenario_multipliers):
    A=np.asarray(alternative_values,float)
    S=np.asarray(scenario_multipliers,float)
    return A[:,None]*S[None,:]

def regret_matrix(payoff_matrix):
    P=np.asarray(payoff_matrix,float)
    best=np.max(P,axis=0)
    return best-P

def minimax_regret(payoff_matrix,alternatives=None):
    regrets=regret_matrix(payoff_matrix)
    max_regret=np.max(regrets,axis=1)
    idx=int(np.argmin(max_regret))
    return {
        "alternative":alternatives[idx] if alternatives else idx,
        "max_regret":float(max_regret[idx]),
        "all_max_regret":max_regret,
    }

def robust_score(payoff_matrix,scenario_probabilities=None,risk_penalty=.5):
    P=np.asarray(payoff_matrix,float)
    if scenario_probabilities is None:
        scenario_probabilities=np.ones(P.shape[1])/P.shape[1]
    probs=np.asarray(scenario_probabilities,float)
    mean=P@probs
    downside=np.std(P,axis=1)
    return mean-risk_penalty*downside

def scenario_table(alternatives,scenarios,payoff_matrix):
    return pd.DataFrame(payoff_matrix,index=alternatives,columns=scenarios)
