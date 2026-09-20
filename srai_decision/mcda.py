"""Multi-criteria decision analysis utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

def normalize_matrix(matrix,benefit_flags):
    X=np.asarray(matrix,float)
    out=np.zeros_like(X,dtype=float)
    for j,benefit in enumerate(benefit_flags):
        col=X[:,j]
        lo,hi=np.min(col),np.max(col)
        if benefit:
            out[:,j]=(col-lo)/(hi-lo+1e-12)
        else:
            out[:,j]=(hi-col)/(hi-lo+1e-12)
    return out

def weighted_sum(matrix,weights,benefit_flags):
    normalized=normalize_matrix(matrix,benefit_flags)
    scores=normalized@np.asarray(weights,float)
    return scores

def rank_alternatives(alternatives,scores):
    rows=[{"alternative":a,"score":float(s)} for a,s in zip(alternatives,scores)]
    return sorted(rows,key=lambda x:x["score"],reverse=True)

def topsis(matrix,weights,benefit_flags):
    X=np.asarray(matrix,float)
    norm=X/np.sqrt((X**2).sum(axis=0))
    weighted=norm*np.asarray(weights,float)
    ideal_best=[]
    ideal_worst=[]
    for j,benefit in enumerate(benefit_flags):
        ideal_best.append(np.max(weighted[:,j]) if benefit else np.min(weighted[:,j]))
        ideal_worst.append(np.min(weighted[:,j]) if benefit else np.max(weighted[:,j]))
    best=np.asarray(ideal_best); worst=np.asarray(ideal_worst)
    d_best=np.sqrt(((weighted-best)**2).sum(axis=1))
    d_worst=np.sqrt(((weighted-worst)**2).sum(axis=1))
    return d_worst/(d_best+d_worst+1e-12)

def sensitivity_weights(matrix,weight_sets,benefit_flags,alternatives):
    rows=[]
    for i,weights in enumerate(weight_sets):
        scores=weighted_sum(matrix,weights,benefit_flags)
        ranking=rank_alternatives(alternatives,scores)
        rows.append({"scenario":i,"winner":ranking[0]["alternative"],"scores":scores})
    return rows
