"""Entropy and information-theory utilities."""
from __future__ import annotations
import numpy as np

def _probabilities(p):
    values=np.asarray(p,dtype=float)
    if values.ndim!=1 or values.size==0: raise ValueError("Probabilities must be a non-empty vector.")
    if np.any(values<0) or not np.isclose(values.sum(),1.0): raise ValueError("Probabilities must be non-negative and sum to one.")
    return values

def entropy(probabilities,base=2.0):
    if base<=0 or np.isclose(base,1.0): raise ValueError("base must be positive and not equal to one.")
    p=_probabilities(probabilities); positive=p[p>0]
    return float(-np.sum(positive*np.log(positive))/np.log(base))

def cross_entropy(p,q,base=2.0):
    if base<=0 or np.isclose(base,1.0): raise ValueError("base must be positive and not equal to one.")
    p=_probabilities(p); q=_probabilities(q)
    if p.shape!=q.shape: raise ValueError("p and q must have identical shapes.")
    if np.any((p>0)&(q==0)): return float("inf")
    mask=p>0
    return float(-np.sum(p[mask]*np.log(q[mask]))/np.log(base))

def kl_divergence(p,q,base=2.0):
    ce=cross_entropy(p,q,base=base)
    return ce if np.isinf(ce) else float(ce-entropy(p,base=base))

def joint_entropy(joint,base=2.0):
    table=np.asarray(joint,dtype=float)
    if table.ndim!=2 or np.any(table<0) or not np.isclose(table.sum(),1.0): raise ValueError("joint must be a non-negative matrix summing to one.")
    positive=table[table>0]
    return float(-np.sum(positive*np.log(positive))/np.log(base))

def marginal_probabilities(joint):
    table=np.asarray(joint,dtype=float)
    if table.ndim!=2 or np.any(table<0) or not np.isclose(table.sum(),1.0): raise ValueError("joint must be a non-negative matrix summing to one.")
    return table.sum(axis=1),table.sum(axis=0)

def conditional_entropy(joint,condition_on="columns",base=2.0):
    table=np.asarray(joint,dtype=float); row,col=marginal_probabilities(table); h=joint_entropy(table,base=base)
    if condition_on=="columns": return float(h-entropy(col,base=base))
    if condition_on=="rows": return float(h-entropy(row,base=base))
    raise ValueError("condition_on must be 'columns' or 'rows'.")

def mutual_information(joint,base=2.0):
    table=np.asarray(joint,dtype=float); row,col=marginal_probabilities(table); value=0.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            pij=table[i,j]
            if pij>0: value+=pij*np.log(pij/(row[i]*col[j]))
    return float(value/np.log(base))

def binary_entropy(p,base=2.0):
    if not 0<=p<=1: raise ValueError("p must lie in [0,1].")
    return entropy([p,1-p],base=base)

def information_gain(parent_probabilities,weighted_children,base=2.0):
    parent=entropy(parent_probabilities,base=base); remaining=0.0; total=0.0
    for weight,child in weighted_children:
        if weight<0: raise ValueError("Weights must be non-negative.")
        remaining+=weight*entropy(child,base=base); total+=weight
    if not np.isclose(total,1.0): raise ValueError("Child weights must sum to one.")
    return float(parent-remaining)
