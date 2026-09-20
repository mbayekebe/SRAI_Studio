"""Deep-learning loss functions."""
from __future__ import annotations
import numpy as np

def mse_loss(y_true,y_pred):
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float)
    return float(np.mean((y-p)**2))

def binary_cross_entropy(y_true,y_prob):
    y=np.asarray(y_true,float); p=np.clip(np.asarray(y_prob,float),1e-12,1-1e-12)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def categorical_cross_entropy(y_true,y_prob):
    y=np.asarray(y_true)
    p=np.clip(np.asarray(y_prob,float),1e-12,1)
    if y.ndim==1:
        return float(-np.mean(np.log(p[np.arange(len(y)),y.astype(int)])))
    return float(-np.mean(np.sum(y*np.log(p),axis=1)))
