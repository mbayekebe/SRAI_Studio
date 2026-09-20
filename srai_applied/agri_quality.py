"""Agricultural data quality checks."""
from __future__ import annotations
import numpy as np
import pandas as pd

def completeness_report(df):
    return pd.DataFrame({
        "column":df.columns,
        "missing_count":[int(df[c].isna().sum()) for c in df.columns],
        "missing_rate":[float(df[c].isna().mean()) for c in df.columns],
    })

def range_check(series,minimum=None,maximum=None):
    s=pd.to_numeric(series,errors="coerce")
    mask=pd.Series(False,index=s.index)
    if minimum is not None:
        mask|=s<minimum
    if maximum is not None:
        mask|=s>maximum
    return mask

def consistency_checks(df):
    results={}
    if {"production","area"}.issubset(df.columns):
        results["negative_production"]=int((df["production"]<0).sum())
        results["negative_area"]=int((df["area"]<0).sum())
        results["production_without_area"]=int(((df["production"]>0)&(df["area"]<=0)).sum())
    if {"male","female","total"}.issubset(df.columns):
        results["sex_total_mismatch"]=int(((df["male"]+df["female"])!=df["total"]).sum())
    return results

def quality_score(df,critical_columns=None):
    critical_columns=critical_columns or list(df.columns)
    completeness=1-float(df[critical_columns].isna().mean().mean())
    uniqueness=1-float(df.duplicated().mean())
    return {
        "completeness":completeness,
        "uniqueness":uniqueness,
        "overall":float((completeness+uniqueness)/2),
    }

def outlier_zscore(series,threshold=3.0):
    s=pd.to_numeric(series,errors="coerce")
    z=(s-s.mean())/s.std(ddof=0)
    return z.abs()>threshold
