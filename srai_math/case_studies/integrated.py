"""Integrated mathematical case-study helpers."""
from __future__ import annotations
import numpy as np

def normalized_scores(X):
    X=np.asarray(X,dtype=float)
    mean=X.mean(axis=0)
    std=X.std(axis=0,ddof=0)
    return (X-mean)/std

def weighted_score(X,weights):
    X=np.asarray(X,dtype=float)
    w=np.asarray(weights,dtype=float)
    if X.shape[1]!=w.size:
        raise ValueError("weight dimension mismatch.")
    return X@w

def scenario_loss(probabilities,losses):
    p=np.asarray(probabilities,dtype=float)
    l=np.asarray(losses,dtype=float)
    if p.shape!=l.shape or not np.isclose(p.sum(),1):
        raise ValueError("Invalid probabilities/losses.")
    return float(p@l)

def portfolio_risk(weights,covariance):
    w=np.asarray(weights,dtype=float)
    C=np.asarray(covariance,dtype=float)
    return float(np.sqrt(w@C@w))

def rank_options(scores,labels=None):
    scores=np.asarray(scores,dtype=float)
    order=np.argsort(scores)[::-1]
    if labels is None:
        return order
    return [(labels[i],float(scores[i])) for i in order]
