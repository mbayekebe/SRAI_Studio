"""Basic preprocessing utilities."""
from __future__ import annotations
import numpy as np

class StandardScaler:
    def fit(self,X):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2: raise ValueError("X must be a matrix.")
        self.mean_=X.mean(axis=0)
        self.scale_=X.std(axis=0,ddof=0)
        self.scale_[np.isclose(self.scale_,0)]=1.0
        return self
    def transform(self,X):
        return (np.asarray(X,dtype=float)-self.mean_)/self.scale_
    def fit_transform(self,X):
        return self.fit(X).transform(X)

def train_test_split(X,y,test_size=0.2,seed=42,shuffle=True):
    X=np.asarray(X); y=np.asarray(y)
    if len(X)!=len(y): raise ValueError("X and y length mismatch.")
    n=len(X); n_test=max(1,int(round(n*test_size)))
    idx=np.arange(n)
    if shuffle:
        rng=np.random.default_rng(seed); rng.shuffle(idx)
    test_idx=idx[:n_test]; train_idx=idx[n_test:]
    return X[train_idx],X[test_idx],y[train_idx],y[test_idx]

def add_intercept(X):
    X=np.asarray(X,dtype=float)
    if X.ndim==1: X=X[:,None]
    return np.column_stack([np.ones(X.shape[0]),X])
