"""Core agricultural analytics."""
from __future__ import annotations
import pandas as pd
import numpy as np

def yield_per_hectare(df,production_col="production",area_col="area"):
    out=df.copy()
    out["yield_per_hectare"]=np.where(
        out[area_col]>0,
        out[production_col]/out[area_col],
        np.nan,
    )
    return out

def aggregate_crop_performance(df,group_cols=("region","crop")):
    grouped=df.groupby(list(group_cols),dropna=False).agg(
        area=("area","sum"),
        production=("production","sum"),
    ).reset_index()
    grouped["yield_per_hectare"]=grouped["production"]/grouped["area"].replace(0,np.nan)
    return grouped

def production_index(series,base_periods=3):
    s=pd.Series(series,dtype=float)
    base=s.iloc[:base_periods].mean()
    return 100*s/base if base else s*np.nan

def food_security_risk_score(price_change,rainfall_anomaly,production_change):
    price=np.clip(np.asarray(price_change,float),-1,1)
    rain=np.clip(np.asarray(rainfall_anomaly,float),-1,1)
    prod=np.clip(np.asarray(production_change,float),-1,1)
    raw=.4*price-.3*rain-.3*prod
    return np.clip((raw+1)/2,0,1)

def classify_risk(score):
    s=float(score)
    if s>=.75: return "critical"
    if s>=.5: return "high"
    if s>=.25: return "medium"
    return "low"
