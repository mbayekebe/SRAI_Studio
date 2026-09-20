"""AI portfolio and value-management helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd

def use_case_score(value,feasibility,data_readiness,risk,time_to_value):
    return float(.30*value+.20*feasibility+.20*data_readiness-.15*risk+.15*(1-time_to_value))

def prioritize_use_cases(use_cases):
    rows=[]
    for item in use_cases:
        score=use_case_score(
            item["value"],item["feasibility"],item["data_readiness"],
            item["risk"],item["time_to_value"],
        )
        rows.append({**item,"priority_score":score})
    return pd.DataFrame(rows).sort_values("priority_score",ascending=False)

def portfolio_balance(frame,category_col="category"):
    return frame[category_col].value_counts(normalize=True).rename("share").reset_index()

def expected_portfolio_value(values,probabilities,costs):
    v=np.asarray(values,float)
    p=np.asarray(probabilities,float)
    c=np.asarray(costs,float)
    return float(np.sum(v*p-c))
