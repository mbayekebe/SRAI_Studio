"""Forecasting models and metrics."""
from __future__ import annotations
import numpy as np
from .timeseries import lag_matrix

class AutoregressiveModel:
    def __init__(self,lags=1):
        self.lags=lags
    def fit(self,series):
        x=np.asarray(series,dtype=float)
        X,y=lag_matrix(x,self.lags)
        A=np.column_stack([np.ones(len(X)),X])
        w=np.linalg.pinv(A)@y
        self.intercept_=float(w[0]); self.coef_=w[1:]
        self.history_=x.copy()
        return self
    def forecast(self,horizon):
        history=list(self.history_)
        out=[]
        for _ in range(horizon):
            features=np.array(history[-self.lags:][::-1],dtype=float)
            value=self.intercept_+features@self.coef_
            history.append(float(value)); out.append(float(value))
        return np.array(out)

class ExponentialSmoothing:
    def __init__(self,alpha=0.3):
        self.alpha=alpha
    def fit(self,series):
        x=np.asarray(series,dtype=float)
        level=x[0]
        fitted=[level]
        for value in x[1:]:
            level=self.alpha*value+(1-self.alpha)*level
            fitted.append(level)
        self.level_=float(level); self.fitted_=np.asarray(fitted); return self
    def forecast(self,horizon):
        return np.full(horizon,self.level_)

def mean_absolute_percentage_error(y_true,y_pred):
    y=np.asarray(y_true,dtype=float); p=np.asarray(y_pred,dtype=float)
    mask=np.abs(y)>1e-12
    return float(np.mean(np.abs((y[mask]-p[mask])/y[mask]))*100)

def symmetric_mape(y_true,y_pred):
    y=np.asarray(y_true,dtype=float); p=np.asarray(y_pred,dtype=float)
    denom=np.abs(y)+np.abs(p)
    mask=denom>1e-12
    return float(np.mean(2*np.abs(y[mask]-p[mask])/denom[mask])*100)

def mean_absolute_scaled_error(y_true,y_pred,training_series,seasonality=1):
    y=np.asarray(y_true,dtype=float); p=np.asarray(y_pred,dtype=float)
    train=np.asarray(training_series,dtype=float)
    scale=np.mean(np.abs(train[seasonality:]-train[:-seasonality]))
    return float(np.mean(np.abs(y-p))/scale)

def walk_forward_validate(model_factory,series,initial_train,horizon=1):
    x=np.asarray(series,dtype=float)
    predictions=[]; actuals=[]
    for start in range(initial_train,len(x)-horizon+1,horizon):
        train=x[:start]
        actual=x[start:start+horizon]
        model=model_factory().fit(train)
        pred=model.forecast(len(actual))
        predictions.extend(pred); actuals.extend(actual)
    return np.asarray(actuals),np.asarray(predictions)
