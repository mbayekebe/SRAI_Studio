"""Monte Carlo simulation for decision analysis."""
from __future__ import annotations
import numpy as np

def monte_carlo(function,n_simulations=10000,seed=42):
    rng=np.random.default_rng(seed)
    values=np.asarray([function(rng) for _ in range(n_simulations)],float)
    return values

def simulation_summary(values):
    x=np.asarray(values,float)
    return {
        "mean":float(np.mean(x)),
        "std":float(np.std(x,ddof=1)),
        "p05":float(np.quantile(x,.05)),
        "p50":float(np.quantile(x,.50)),
        "p95":float(np.quantile(x,.95)),
        "probability_loss":float(np.mean(x<0)),
    }

def compare_simulated_alternatives(simulations):
    summaries={name:simulation_summary(values) for name,values in simulations.items()}
    winner=max(summaries,key=lambda name:summaries[name]["mean"])
    return {"winner":winner,"summaries":summaries}

def risk_adjusted_value(values,risk_aversion=.5):
    x=np.asarray(values,float)
    return float(np.mean(x)-risk_aversion*np.std(x,ddof=1))
