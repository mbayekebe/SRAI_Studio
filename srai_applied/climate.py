"""Rainfall and drought analytics."""
from __future__ import annotations
import numpy as np
import pandas as pd

def rainfall_anomaly(rainfall,climatology):
    return np.asarray(rainfall,float)-np.asarray(climatology,float)

def rainfall_anomaly_percent(rainfall,climatology):
    r=np.asarray(rainfall,float); c=np.asarray(climatology,float)
    return 100*(r-c)/np.where(c==0,np.nan,c)

def standardized_precipitation_index(rainfall):
    x=np.asarray(rainfall,float)
    return (x-x.mean())/x.std(ddof=0)

def cumulative_rainfall(rainfall):
    return np.cumsum(np.asarray(rainfall,float))

def drought_class(spi):
    x=float(spi)
    if x<=-2: return "extreme"
    if x<=-1.5: return "severe"
    if x<=-1: return "moderate"
    if x<0: return "mild"
    return "normal_or_wet"

def onset_date(rainfall,threshold=20,window=3):
    x=np.asarray(rainfall,float)
    for i in range(len(x)-window+1):
        if x[i:i+window].sum()>=threshold:
            return i
    return None
