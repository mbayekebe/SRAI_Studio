"""Regression and classification metrics."""
from __future__ import annotations
import numpy as np

def mean_squared_error(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=float); y_pred=np.asarray(y_pred,dtype=float)
    return float(np.mean((y_true-y_pred)**2))

def root_mean_squared_error(y_true,y_pred):
    return float(np.sqrt(mean_squared_error(y_true,y_pred)))

def mean_absolute_error(y_true,y_pred):
    return float(np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred))))

def r2_score(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=float); y_pred=np.asarray(y_pred,dtype=float)
    ss_res=np.sum((y_true-y_pred)**2)
    ss_tot=np.sum((y_true-y_true.mean())**2)
    return float(1-ss_res/ss_tot)

def confusion_matrix(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=int); y_pred=np.asarray(y_pred,dtype=int)
    tn=np.sum((y_true==0)&(y_pred==0)); fp=np.sum((y_true==0)&(y_pred==1))
    fn=np.sum((y_true==1)&(y_pred==0)); tp=np.sum((y_true==1)&(y_pred==1))
    return np.array([[tn,fp],[fn,tp]],dtype=int)

def accuracy_score(y_true,y_pred):
    return float(np.mean(np.asarray(y_true)==np.asarray(y_pred)))

def precision_score(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred); tp=cm[1,1]; fp=cm[0,1]
    return float(tp/(tp+fp)) if tp+fp else 0.0

def recall_score(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred); tp=cm[1,1]; fn=cm[1,0]
    return float(tp/(tp+fn)) if tp+fn else 0.0

def f1_score(y_true,y_pred):
    p=precision_score(y_true,y_pred); r=recall_score(y_true,y_pred)
    return float(2*p*r/(p+r)) if p+r else 0.0

def log_loss(y_true,prob):
    y=np.asarray(y_true,dtype=float); p=np.clip(np.asarray(prob,dtype=float),1e-12,1-1e-12)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
