"""Causal inference utilities."""
from __future__ import annotations
import numpy as np

def difference_in_means(outcome,treatment):
    y=np.asarray(outcome,dtype=float); t=np.asarray(treatment,dtype=int)
    return float(y[t==1].mean()-y[t==0].mean())

def standardized_mean_difference(X,treatment):
    X=np.asarray(X,dtype=float); t=np.asarray(treatment,dtype=int)
    m1=X[t==1].mean(axis=0); m0=X[t==0].mean(axis=0)
    v1=X[t==1].var(axis=0,ddof=1); v0=X[t==0].var(axis=0,ddof=1)
    pooled=np.sqrt((v1+v0)/2)
    return (m1-m0)/pooled

class PropensityScoreLogit:
    def __init__(self,learning_rate=0.1,max_iter=2000):
        self.learning_rate=learning_rate; self.max_iter=max_iter
    def fit(self,X,treatment):
        X=np.asarray(X,dtype=float); t=np.asarray(treatment,dtype=float)
        A=np.column_stack([np.ones(len(X)),X])
        w=np.zeros(A.shape[1])
        for _ in range(self.max_iter):
            z=np.clip(A@w,-500,500)
            p=1/(1+np.exp(-z))
            grad=A.T@(p-t)/len(t)
            w-=self.learning_rate*grad
        self.intercept_=float(w[0]); self.coef_=w[1:]; return self
    def predict_proba(self,X):
        z=np.clip(np.asarray(X,dtype=float)@self.coef_+self.intercept_,-500,500)
        return 1/(1+np.exp(-z))

def inverse_probability_weighted_ate(outcome,treatment,propensity,clip=1e-3):
    y=np.asarray(outcome,dtype=float)
    t=np.asarray(treatment,dtype=float)
    p=np.clip(np.asarray(propensity,dtype=float),clip,1-clip)
    return float(np.mean(t*y/p-(1-t)*y/(1-p)))

def propensity_score_matching(outcome,treatment,propensity):
    y=np.asarray(outcome,dtype=float); t=np.asarray(treatment,dtype=int)
    p=np.asarray(propensity,dtype=float)
    treated=np.where(t==1)[0]; control=np.where(t==0)[0]
    diffs=[]
    for i in treated:
        j=control[np.argmin(np.abs(p[control]-p[i]))]
        diffs.append(y[i]-y[j])
    return float(np.mean(diffs))

def randomized_difference_in_means(outcome,treatment):
    return difference_in_means(outcome,treatment)
