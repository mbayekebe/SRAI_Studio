"""Treatment-effect and uplift modeling utilities."""
from __future__ import annotations
import numpy as np
from .linear_models import LinearRegression

class TLearner:
    def __init__(self,model_factory=lambda:LinearRegression()):
        self.model_factory=model_factory
    def fit(self,X,treatment,outcome):
        X=np.asarray(X,dtype=float); t=np.asarray(treatment,dtype=int); y=np.asarray(outcome,dtype=float)
        self.model_t_=self.model_factory().fit(X[t==1],y[t==1])
        self.model_c_=self.model_factory().fit(X[t==0],y[t==0])
        return self
    def predict_cate(self,X):
        X=np.asarray(X,dtype=float)
        return self.model_t_.predict(X)-self.model_c_.predict(X)

class SLearner:
    def __init__(self,model_factory=lambda:LinearRegression()):
        self.model_factory=model_factory
    def fit(self,X,treatment,outcome):
        X=np.asarray(X,dtype=float); t=np.asarray(treatment,dtype=float)[:,None]
        self.model_=self.model_factory().fit(np.column_stack([X,t]),outcome)
        return self
    def predict_cate(self,X):
        X=np.asarray(X,dtype=float)
        y1=self.model_.predict(np.column_stack([X,np.ones(len(X))]))
        y0=self.model_.predict(np.column_stack([X,np.zeros(len(X))]))
        return y1-y0

def policy_value(outcome,treatment,recommended_treatment):
    y=np.asarray(outcome,dtype=float)
    t=np.asarray(treatment,dtype=int)
    r=np.asarray(recommended_treatment,dtype=int)
    mask=t==r
    return float(np.mean(y[mask])) if np.any(mask) else float("nan")

def uplift_by_quantile(cate,observed_outcome,treatment,n_bins=5):
    c=np.asarray(cate,dtype=float); y=np.asarray(observed_outcome,dtype=float); t=np.asarray(treatment,dtype=int)
    order=np.argsort(c)[::-1]
    bins=np.array_split(order,n_bins)
    rows=[]
    for i,b in enumerate(bins,1):
        uplift=y[b][t[b]==1].mean()-y[b][t[b]==0].mean() if np.any(t[b]==1) and np.any(t[b]==0) else np.nan
        rows.append((i,float(np.mean(c[b])),float(uplift)))
    return rows
