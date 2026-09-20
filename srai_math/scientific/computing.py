"""Scientific computing, vectorization, and sparse utilities."""
from __future__ import annotations
import numpy as np
from scipy import sparse

def standardize_columns(X):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2:
        raise ValueError("X must be a matrix.")
    mean=X.mean(axis=0)
    std=X.std(axis=0,ddof=0)
    if np.any(np.isclose(std,0)):
        raise ValueError("Constant columns cannot be standardized.")
    return (X-mean)/std,mean,std

def pairwise_squared_distances(X):
    X=np.asarray(X,dtype=float)
    norms=np.sum(X**2,axis=1,keepdims=True)
    D=norms+norms.T-2*X@X.T
    return np.maximum(D,0.0)

def moving_average(x,window):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or window<=0 or window>x.size:
        raise ValueError("Invalid inputs.")
    c=np.cumsum(np.insert(x,0,0.0))
    return (c[window:]-c[:-window])/window

def to_csr(X):
    return sparse.csr_matrix(np.asarray(X,dtype=float))

def sparse_matvec(matrix,vector):
    return np.asarray(matrix@np.asarray(vector,dtype=float)).ravel()

def chunked_mean(X,chunk_size):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or chunk_size<=0:
        raise ValueError("Invalid inputs.")
    total=np.zeros(X.shape[1])
    count=0
    for start in range(0,X.shape[0],chunk_size):
        chunk=X[start:start+chunk_size]
        total+=chunk.sum(axis=0)
        count+=chunk.shape[0]
    return total/count
