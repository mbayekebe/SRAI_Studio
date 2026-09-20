"""Clustering algorithms."""
from __future__ import annotations
import numpy as np

class KMeans:
    def __init__(self,n_clusters=3,max_iter=300,tol=1e-4,seed=42):
        self.n_clusters=n_clusters; self.max_iter=max_iter; self.tol=tol; self.seed=seed
    def fit(self,X):
        X=np.asarray(X,float)
        rng=np.random.default_rng(self.seed)
        idx=rng.choice(len(X),self.n_clusters,replace=False)
        centers=X[idx].copy()
        self.inertia_history_=[]
        for _ in range(self.max_iter):
            d=((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
            labels=np.argmin(d,axis=1)
            inertia=float(np.sum(np.min(d,axis=1)))
            self.inertia_history_.append(inertia)
            new_centers=np.array([
                X[labels==k].mean(axis=0) if np.any(labels==k) else centers[k]
                for k in range(self.n_clusters)
            ])
            if np.linalg.norm(new_centers-centers)<=self.tol:
                centers=new_centers; break
            centers=new_centers
        self.cluster_centers_=centers; self.labels_=labels; self.inertia_=self.inertia_history_[-1]
        return self
    def predict(self,X):
        X=np.asarray(X,float)
        d=((X[:,None,:]-self.cluster_centers_[None,:,:])**2).sum(axis=2)
        return np.argmin(d,axis=1)

def silhouette_score(X,labels):
    X=np.asarray(X,float); labels=np.asarray(labels,int)
    unique=np.unique(labels)
    if len(unique)<2: raise ValueError("At least two clusters required.")
    D=np.sqrt(np.maximum(((X[:,None,:]-X[None,:,:])**2).sum(axis=2),0))
    values=[]
    for i in range(len(X)):
        same=labels==labels[i]
        same[i]=False
        a=D[i,same].mean() if np.any(same) else 0.0
        b=min(D[i,labels==c].mean() for c in unique if c!=labels[i])
        values.append((b-a)/max(a,b) if max(a,b)>0 else 0.0)
    return float(np.mean(values))

class AgglomerativeClustering:
    def __init__(self,n_clusters=2):
        self.n_clusters=n_clusters
    def fit(self,X):
        X=np.asarray(X,float)
        clusters=[[i] for i in range(len(X))]
        while len(clusters)>self.n_clusters:
            best=None; best_dist=np.inf
            for i in range(len(clusters)):
                for j in range(i+1,len(clusters)):
                    a=X[clusters[i]].mean(axis=0); b=X[clusters[j]].mean(axis=0)
                    dist=np.linalg.norm(a-b)
                    if dist<best_dist:
                        best_dist=dist; best=(i,j)
            i,j=best
            clusters[i]=clusters[i]+clusters[j]
            del clusters[j]
        labels=np.empty(len(X),dtype=int)
        for k,cluster in enumerate(clusters):
            labels[cluster]=k
        self.labels_=labels
        return self
