"""Graph-based machine-learning utilities."""
from __future__ import annotations
import numpy as np

def normalize_adjacency(A,add_self_loops=True):
    A=np.asarray(A,float)
    if add_self_loops:
        A=A+np.eye(A.shape[0])
    degree=A.sum(axis=1)
    inv_sqrt=np.zeros_like(degree)
    mask=degree>0
    inv_sqrt[mask]=1/np.sqrt(degree[mask])
    D=np.diag(inv_sqrt)
    return D@A@D

def graph_convolution(A,X,W):
    return normalize_adjacency(A)@np.asarray(X,float)@np.asarray(W,float)

def label_propagation(A,labels,labeled_mask,alpha=0.9,max_iter=200):
    A_norm=normalize_adjacency(A)
    labels=np.asarray(labels,int)
    mask=np.asarray(labeled_mask,bool)
    classes=np.max(labels[mask])+1
    Y=np.zeros((len(labels),classes))
    Y[np.where(mask)[0],labels[mask]]=1.0
    F=Y.copy()
    for _ in range(max_iter):
        F=alpha*A_norm@F+(1-alpha)*Y
        F[mask]=Y[mask]
    return F

def degree_centrality(A):
    A=np.asarray(A,float)
    return A.sum(axis=1)/(A.shape[0]-1)

def pagerank(A,damping=0.85,max_iter=200,tol=1e-12):
    A=np.asarray(A,float)
    n=A.shape[0]
    row_sum=A.sum(axis=1,keepdims=True)
    P=np.divide(A,row_sum,out=np.full_like(A,1/n),where=row_sum>0)
    rank=np.full(n,1/n)
    for _ in range(max_iter):
        new=(1-damping)/n+damping*P.T@rank
        if np.linalg.norm(new-rank,1)<tol:
            rank=new; break
        rank=new
    return rank
