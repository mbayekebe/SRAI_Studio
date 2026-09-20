"""Time-series utilities."""
from __future__ import annotations
import numpy as np

def lag_matrix(series,lags):
    x=np.asarray(series,dtype=float)
    if x.ndim!=1 or lags<=0 or len(x)<=lags:
        raise ValueError("Invalid series or lags.")
    X=np.column_stack([x[lags-i-1:len(x)-i-1] for i in range(lags)])
    y=x[lags:]
    return X,y

def difference(series,periods=1):
    x=np.asarray(series,dtype=float)
    if periods<=0 or periods>=len(x):
        raise ValueError("Invalid periods.")
    return x[periods:]-x[:-periods]

def seasonal_naive_forecast(series,horizon,season_length):
    x=np.asarray(series,dtype=float)
    if horizon<=0 or season_length<=0 or len(x)<season_length:
        raise ValueError("Invalid inputs.")
    pattern=x[-season_length:]
    return np.array([pattern[i%season_length] for i in range(horizon)],dtype=float)

def rolling_mean(series,window):
    x=np.asarray(series,dtype=float)
    if window<=0 or window>len(x):
        raise ValueError("Invalid window.")
    c=np.cumsum(np.insert(x,0,0.0))
    return (c[window:]-c[:-window])/window

def autocorrelation(series,max_lag):
    x=np.asarray(series,dtype=float)
    x=x-x.mean()
    denom=np.sum(x*x)
    return np.array([1.0]+[
        np.sum(x[:-lag]*x[lag:])/denom for lag in range(1,max_lag+1)
    ])

def time_series_split(n_samples,n_splits=5,test_size=None):
    if n_samples<=n_splits:
        raise ValueError("Not enough samples.")
    if test_size is None:
        test_size=max(1,n_samples//(n_splits+1))
    for i in range(n_splits):
        test_end=n_samples-(n_splits-i-1)*test_size
        test_start=test_end-test_size
        train=np.arange(0,test_start)
        test=np.arange(test_start,test_end)
        yield train,test
