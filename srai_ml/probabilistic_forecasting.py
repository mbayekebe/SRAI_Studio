"""Probabilistic forecasting utilities."""
from __future__ import annotations
import numpy as np
from scipy import stats

def pinball_loss(y_true,quantile_forecast,quantile):
    y=np.asarray(y_true,dtype=float); q=np.asarray(quantile_forecast,dtype=float)
    e=y-q
    return float(np.mean(np.maximum(quantile*e,(quantile-1)*e)))

def prediction_interval_coverage(y_true,lower,upper):
    y=np.asarray(y_true,dtype=float)
    return float(np.mean((y>=np.asarray(lower))&(y<=np.asarray(upper))))

def interval_width(lower,upper):
    return float(np.mean(np.asarray(upper)-np.asarray(lower)))

def gaussian_prediction_interval(mean,std,level=0.95):
    z=stats.norm.ppf(0.5+level/2)
    mean=np.asarray(mean,dtype=float)
    std=np.asarray(std,dtype=float)
    return mean-z*std,mean+z*std

def bootstrap_forecast_intervals(point_forecast,residuals,horizon,repetitions=2000,level=0.95,seed=42):
    point=np.asarray(point_forecast,dtype=float)
    residuals=np.asarray(residuals,dtype=float)
    rng=np.random.default_rng(seed)
    simulations=np.empty((repetitions,horizon))
    for i in range(repetitions):
        simulations[i]=point[:horizon]+rng.choice(residuals,size=horizon,replace=True)
    alpha=(1-level)/2
    return np.quantile(simulations,alpha,axis=0),np.quantile(simulations,1-alpha,axis=0)

def crps_ensemble(y_true,ensemble):
    y=np.asarray(y_true,dtype=float)
    E=np.asarray(ensemble,dtype=float)
    term1=np.mean(np.abs(E-y[None,:]),axis=0)
    term2=0.5*np.mean(np.abs(E[:,None,:]-E[None,:,:]),axis=(0,1))
    return float(np.mean(term1-term2))
