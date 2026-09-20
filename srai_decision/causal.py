"""Causal decision-analysis utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

def difference_in_means(outcome,treatment):
    y=np.asarray(outcome,float)
    t=np.asarray(treatment,int)
    return float(y[t==1].mean()-y[t==0].mean())

def inverse_probability_weighted_ate(outcome,treatment,propensity):
    y=np.asarray(outcome,float)
    t=np.asarray(treatment,int)
    p=np.clip(np.asarray(propensity,float),1e-6,1-1e-6)
    treated=np.mean(t*y/p)
    control=np.mean((1-t)*y/(1-p))
    return float(treated-control)

def heterogeneous_treatment_effect(frame,outcome_col,treatment_col,group_col):
    rows=[]
    for group,part in frame.groupby(group_col):
        effect=difference_in_means(part[outcome_col],part[treatment_col])
        rows.append({"group":group,"effect":effect,"n":len(part)})
    return pd.DataFrame(rows)

def policy_value(treatment_effects,costs,eligible=None):
    effects=np.asarray(treatment_effects,float)
    costs=np.asarray(costs,float)
    mask=np.ones(len(effects),dtype=bool) if eligible is None else np.asarray(eligible,bool)
    net=np.where(mask,effects-costs,0)
    return {"net_values":net,"total_value":float(net.sum())}
