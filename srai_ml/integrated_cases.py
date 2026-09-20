"""Integrated machine-learning case-study helpers."""
from __future__ import annotations
import numpy as np

def cost_sensitive_decision(probabilities,threshold):
    return (np.asarray(probabilities,float)>=threshold).astype(int)

def rank_by_expected_value(probability,benefit,cost):
    p=np.asarray(probability,float)
    return p*benefit-(1-p)*cost

def fairness_group_rates(y_true,y_pred,groups):
    y=np.asarray(y_true,int); p=np.asarray(y_pred,int); g=np.asarray(groups)
    result={}
    for group in np.unique(g):
        mask=g==group
        tp=np.sum((p[mask]==1)&(y[mask]==1))
        fn=np.sum((p[mask]==0)&(y[mask]==1))
        fp=np.sum((p[mask]==1)&(y[mask]==0))
        tn=np.sum((p[mask]==0)&(y[mask]==0))
        result[str(group)]={
            "TPR":float(tp/(tp+fn)) if tp+fn else 0.0,
            "FPR":float(fp/(fp+tn)) if fp+tn else 0.0,
            "selection_rate":float(np.mean(p[mask]==1)),
        }
    return result

def drift_population_stability_index(reference,current,bins=10):
    r=np.asarray(reference,float); c=np.asarray(current,float)
    edges=np.quantile(r,np.linspace(0,1,bins+1))
    edges[0]-=1e-9; edges[-1]+=1e-9
    rh=np.histogram(r,bins=edges)[0]/len(r)
    ch=np.histogram(c,bins=edges)[0]/len(c)
    rh=np.clip(rh,1e-6,None); ch=np.clip(ch,1e-6,None)
    return float(np.sum((ch-rh)*np.log(ch/rh)))
