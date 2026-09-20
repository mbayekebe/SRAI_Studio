"""Prescriptive analytics utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

def recommend_actions(scores,thresholds,actions):
    recommendations=[]
    for score in np.asarray(scores,float):
        selected=actions[-1]
        for threshold,action in zip(thresholds,actions):
            if score<=threshold:
                selected=action
                break
        recommendations.append(selected)
    return recommendations

def resource_allocation(priority_scores,total_budget,minimum_allocations=None):
    scores=np.asarray(priority_scores,float)
    minimum=np.zeros_like(scores) if minimum_allocations is None else np.asarray(minimum_allocations,float)
    remaining=total_budget-minimum.sum()
    if remaining<0:
        raise ValueError("Minimum allocations exceed total budget.")
    weights=scores/scores.sum() if scores.sum()>0 else np.ones_like(scores)/len(scores)
    return minimum+remaining*weights

def intervention_plan(entities,priorities,budget,minimum=None):
    allocations=resource_allocation(priorities,budget,minimum)
    return pd.DataFrame({
        "entity":entities,
        "priority":priorities,
        "allocation":allocations,
    }).sort_values("priority",ascending=False)

def expected_net_benefit(benefits,costs,probabilities=None):
    b=np.asarray(benefits,float)
    c=np.asarray(costs,float)
    p=np.ones_like(b)/len(b) if probabilities is None else np.asarray(probabilities,float)
    return float(np.sum(p*(b-c)))
