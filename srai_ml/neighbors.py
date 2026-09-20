"""Nearest-neighbor models."""
from __future__ import annotations
import numpy as np

def euclidean_distances(X,Y):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    D=np.sum(X**2,axis=1)[:,None]+np.sum(Y**2,axis=1)[None,:]-2*X@Y.T
    return np.sqrt(np.maximum(D,0))

class KNNClassifier:
    def __init__(self,n_neighbors=5,weights="uniform"):
        self.n_neighbors=n_neighbors; self.weights=weights
    def fit(self,X,y):
        self.X_=np.asarray(X,float); self.y_=np.asarray(y,int); return self
    def predict(self,X):
        D=euclidean_distances(np.asarray(X,float),self.X_)
        idx=np.argsort(D,axis=1)[:,:self.n_neighbors]
        out=[]
        for i,row in enumerate(idx):
            labels=self.y_[row]
            if self.weights=="distance":
                w=1/(D[i,row]+1e-12)
                scores=np.bincount(labels,weights=w)
            else:
                scores=np.bincount(labels)
            out.append(np.argmax(scores))
        return np.asarray(out,dtype=int)

class KNNRegressor:
    def __init__(self,n_neighbors=5,weights="uniform"):
        self.n_neighbors=n_neighbors; self.weights=weights
    def fit(self,X,y):
        self.X_=np.asarray(X,float); self.y_=np.asarray(y,float); return self
    def predict(self,X):
        D=euclidean_distances(np.asarray(X,float),self.X_)
        idx=np.argsort(D,axis=1)[:,:self.n_neighbors]
        out=[]
        for i,row in enumerate(idx):
            vals=self.y_[row]
            if self.weights=="distance":
                w=1/(D[i,row]+1e-12); out.append(np.sum(w*vals)/np.sum(w))
            else:
                out.append(np.mean(vals))
        return np.asarray(out)
