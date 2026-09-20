"""Agricultural production forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd

def seasonal_naive(series,horizon,season_length):
    x=np.asarray(series,float)
    pattern=x[-season_length:]
    return np.array([pattern[i%season_length] for i in range(horizon)],float)

def moving_average_forecast(series,horizon,window=3):
    history=list(np.asarray(series,float))
    out=[]
    for _ in range(horizon):
        value=float(np.mean(history[-window:]))
        history.append(value); out.append(value)
    return np.asarray(out)

def trend_forecast(series,horizon):
    x=np.asarray(series,float)
    t=np.arange(len(x))
    slope,intercept=np.polyfit(t,x,1)
    future=np.arange(len(x),len(x)+horizon)
    return intercept+slope*future

def forecast_interval(point_forecast,residual_std,level=0.95):
    z=1.96 if level==0.95 else 1.645
    p=np.asarray(point_forecast,float)
    return p-z*residual_std,p+z*residual_std

def mae(actual,forecast):
    return float(np.mean(np.abs(np.asarray(actual,float)-np.asarray(forecast,float))))

def ensemble_forecast(forecasts,weights=None):
    F=np.asarray(forecasts,float)
    if weights is None:
        weights=np.ones(F.shape[0])/F.shape[0]
    return np.average(F,axis=0,weights=np.asarray(weights,float))
