"""Policy evaluation utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

def cost_effectiveness(costs,effects):
    c=np.asarray(costs,float)
    e=np.asarray(effects,float)
    return np.divide(c,e,out=np.full_like(c,np.inf),where=e!=0)

def incremental_cost_effectiveness(cost_a,effect_a,cost_b,effect_b):
    delta_cost=float(cost_b-cost_a)
    delta_effect=float(effect_b-effect_a)
    return delta_cost/delta_effect if delta_effect!=0 else np.inf

def policy_score(effectiveness,equity,feasibility,sustainability,weights=None):
    values=np.array([effectiveness,equity,feasibility,sustainability],float)
    weights=np.asarray(weights if weights is not None else [.35,.25,.20,.20],float)
    return float(values@weights/weights.sum())

def distributional_impact(frame,group_col,impact_col):
    grouped=frame.groupby(group_col)[impact_col].agg(["mean","sum","count"]).reset_index()
    return grouped

def policy_portfolio_rank(policies):
    rows=[]
    for policy in policies:
        score=policy_score(
            policy["effectiveness"],policy["equity"],
            policy["feasibility"],policy["sustainability"],
            policy.get("weights"),
        )
        rows.append({"policy":policy["name"],"score":score})
    return sorted(rows,key=lambda x:x["score"],reverse=True)
