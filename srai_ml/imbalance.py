"""Imbalanced-learning utilities."""
from __future__ import annotations
import numpy as np

def class_weights(y):
    y=np.asarray(y,int)
    classes,counts=np.unique(y,return_counts=True)
    n=len(y); k=len(classes)
    return {int(c):float(n/(k*count)) for c,count in zip(classes,counts)}

def random_oversample(X,y,seed=42):
    X=np.asarray(X); y=np.asarray(y)
    rng=np.random.default_rng(seed)
    classes,counts=np.unique(y,return_counts=True)
    target=counts.max()
    idx=[]
    for c,count in zip(classes,counts):
        cidx=np.where(y==c)[0]
        selected=rng.choice(cidx,size=target,replace=True)
        idx.extend(selected.tolist())
    idx=np.asarray(idx)
    rng.shuffle(idx)
    return X[idx],y[idx]

def random_undersample(X,y,seed=42):
    X=np.asarray(X); y=np.asarray(y)
    rng=np.random.default_rng(seed)
    classes,counts=np.unique(y,return_counts=True)
    target=counts.min()
    idx=[]
    for c in classes:
        cidx=np.where(y==c)[0]
        idx.extend(rng.choice(cidx,size=target,replace=False).tolist())
    idx=np.asarray(idx)
    rng.shuffle(idx)
    return X[idx],y[idx]

def precision_recall_curve(y_true,scores):
    y=np.asarray(y_true,int); s=np.asarray(scores,float)
    thresholds=np.unique(s)[::-1]
    precision=[]; recall=[]
    positives=np.sum(y==1)
    for t in thresholds:
        pred=s>=t
        tp=np.sum((pred==1)&(y==1))
        fp=np.sum((pred==1)&(y==0))
        precision.append(tp/(tp+fp) if tp+fp else 1.0)
        recall.append(tp/positives if positives else 0.0)
    return np.asarray(precision),np.asarray(recall),thresholds

def average_precision(y_true,scores):
    y=np.asarray(y_true,int); s=np.asarray(scores,float)
    positives=np.sum(y==1)
    if positives == 0:
        return 0.0
    order=np.argsort(-s,kind="stable")
    ranked=y[order]
    precision=np.cumsum(ranked==1)/np.arange(1,len(ranked)+1)
    return float(np.sum(precision[ranked==1])/positives)

def balanced_accuracy(y_true,y_pred):
    y=np.asarray(y_true,int); p=np.asarray(y_pred,int)
    recalls=[]
    for c in np.unique(y):
        mask=y==c
        recalls.append(np.mean(p[mask]==c))
    return float(np.mean(recalls))
