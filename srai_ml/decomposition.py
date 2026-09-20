"""Dimensionality reduction utilities."""
from __future__ import annotations
import numpy as np

class PCA:
    def __init__(self,n_components=None):
        self.n_components=n_components
    def fit(self,X):
        X=np.asarray(X,float)
        self.mean_=X.mean(axis=0)
        Xc=X-self.mean_
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
        maxc=min(X.shape)
        k=maxc if self.n_components is None else self.n_components
        self.components_=Vt[:k]
        variance=(s**2)/(len(X)-1)
        self.explained_variance_=variance[:k]
        self.explained_variance_ratio_=variance[:k]/variance.sum()
        return self
    def transform(self,X):
        return (np.asarray(X,float)-self.mean_)@self.components_.T
    def fit_transform(self,X):
        return self.fit(X).transform(X)
    def inverse_transform(self,Z):
        return np.asarray(Z,float)@self.components_+self.mean_

class RandomProjection:
    def __init__(self,n_components,seed=42):
        self.n_components=n_components; self.seed=seed
    def fit(self,X):
        X=np.asarray(X,float)
        rng=np.random.default_rng(self.seed)
        self.components_=rng.normal(0,1/np.sqrt(self.n_components),
                                    size=(X.shape[1],self.n_components))
        return self
    def transform(self,X):
        return np.asarray(X,float)@self.components_
    def fit_transform(self,X):
        return self.fit(X).transform(X)
