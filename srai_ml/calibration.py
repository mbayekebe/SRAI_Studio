"""Probability calibration and decision-threshold utilities."""
from __future__ import annotations
import numpy as np

def brier_score(y_true,probabilities):
    y=np.asarray(y_true,float); p=np.asarray(probabilities,float)
    return float(np.mean((p-y)**2))

def calibration_curve(y_true,probabilities,n_bins=10):
    y=np.asarray(y_true,float); p=np.asarray(probabilities,float)
    edges=np.linspace(0,1,n_bins+1)
    mean_pred=[]; frac_pos=[]; counts=[]
    for i in range(n_bins):
        mask=(p>=edges[i])&(p<(edges[i+1] if i<n_bins-1 else edges[i+1]+1e-12))
        if np.any(mask):
            mean_pred.append(np.mean(p[mask]))
            frac_pos.append(np.mean(y[mask]))
            counts.append(np.sum(mask))
    return np.asarray(mean_pred),np.asarray(frac_pos),np.asarray(counts)

class PlattCalibrator:
    def __init__(self,learning_rate=0.1,max_iter=2000):
        self.learning_rate=learning_rate; self.max_iter=max_iter
    def fit(self,scores,y):
        x=np.asarray(scores,float); y=np.asarray(y,float)
        a=0.0; b=0.0
        for _ in range(self.max_iter):
            z=np.clip(a*x+b,-500,500)
            p=1/(1+np.exp(-z))
            grad_a=np.mean((p-y)*x)
            grad_b=np.mean(p-y)
            a-=self.learning_rate*grad_a
            b-=self.learning_rate*grad_b
        self.a_=a; self.b_=b
        return self
    def predict_proba(self,scores):
        z=np.clip(self.a_*np.asarray(scores,float)+self.b_,-500,500)
        return 1/(1+np.exp(-z))

def expected_cost(y_true,probabilities,threshold,false_positive_cost=1.0,false_negative_cost=1.0):
    y=np.asarray(y_true,int); p=np.asarray(probabilities,float)
    pred=(p>=threshold).astype(int)
    fp=np.sum((pred==1)&(y==0))
    fn=np.sum((pred==0)&(y==1))
    return float(fp*false_positive_cost+fn*false_negative_cost)

def optimal_threshold(y_true,probabilities,false_positive_cost=1.0,false_negative_cost=1.0):
    candidates=np.unique(np.asarray(probabilities,float))
    costs=[expected_cost(y_true,probabilities,t,false_positive_cost,false_negative_cost) for t in candidates]
    i=int(np.argmin(costs))
    return float(candidates[i]),float(costs[i])
