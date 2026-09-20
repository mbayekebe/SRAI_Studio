"""Crop-risk early-warning system."""
from __future__ import annotations
import numpy as np
import pandas as pd

def normalize_risk(values,low=None,high=None):
    x=np.asarray(values,float)
    low=np.nanmin(x) if low is None else low
    high=np.nanmax(x) if high is None else high
    return np.clip((x-low)/(high-low+1e-12),0,1)

def fuse_risks(drought,price,vegetation,pest,weights=None):
    X=np.column_stack([drought,price,vegetation,pest]).astype(float)
    w=np.asarray(weights if weights is not None else [.3,.25,.25,.2],float)
    return np.clip(X@w/w.sum(),0,1)

def alert_level(score):
    s=float(score)
    if s>=.8: return "emergency"
    if s>=.6: return "alert"
    if s>=.4: return "watch"
    return "normal"

def production_loss_estimate(expected_production,risk_score,max_loss_fraction=.6):
    expected=np.asarray(expected_production,float)
    risk=np.asarray(risk_score,float)
    return expected*risk*max_loss_fraction

def prioritize_interventions(frame,score_col="risk_score",capacity=5):
    df=pd.DataFrame(frame).copy()
    return df.sort_values(score_col,ascending=False).head(capacity)
