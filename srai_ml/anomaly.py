"""Anomaly detection utilities."""
from __future__ import annotations
import numpy as np

class ZScoreDetector:
    def __init__(self,threshold=3.0):
        self.threshold=threshold
    def fit(self,X):
        X=np.asarray(X,float)
        self.mean_=X.mean(axis=0)
        self.std_=X.std(axis=0,ddof=0)
        self.std_[np.isclose(self.std_,0)]=1.0
        return self
    def score_samples(self,X):
        Z=np.abs((np.asarray(X,float)-self.mean_)/self.std_)
        return np.max(Z,axis=1)
    def predict(self,X):
        return (self.score_samples(X)>self.threshold).astype(int)

class MahalanobisDetector:
    def __init__(self,threshold_quantile=0.99,regularization=1e-6):
        self.threshold_quantile=threshold_quantile; self.regularization=regularization
    def fit(self,X):
        X=np.asarray(X,float)
        self.mean_=X.mean(axis=0)
        cov=np.cov(X,rowvar=False)+self.regularization*np.eye(X.shape[1])
        self.inv_cov_=np.linalg.inv(cov)
        scores=self.score_samples(X)
        self.threshold_=float(np.quantile(scores,self.threshold_quantile))
        return self
    def score_samples(self,X):
        D=np.asarray(X,float)-self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i",D,self.inv_cov_,D))
    def predict(self,X):
        return (self.score_samples(X)>self.threshold_).astype(int)

class IsolationLikeDetector:
    def __init__(self,n_estimators=100,subsample=64,seed=42):
        self.n_estimators=n_estimators; self.subsample=subsample; self.seed=seed
    def fit(self,X):
        X=np.asarray(X,float); rng=np.random.default_rng(self.seed)
        self.models_=[]
        for _ in range(self.n_estimators):
            idx=rng.choice(len(X),min(self.subsample,len(X)),replace=False)
            sub=X[idx]
            feature=int(rng.integers(0,X.shape[1]))
            threshold=float(rng.uniform(sub[:,feature].min(),sub[:,feature].max()))
            self.models_.append((feature,threshold))
        return self
    def score_samples(self,X):
        X=np.asarray(X,float)
        rarity=np.zeros(len(X))
        for feature,threshold in self.models_:
            left=X[:,feature]<=threshold
            p=min(np.mean(left),1-np.mean(left))
            rarity+=np.where(left,1-np.mean(left),np.mean(left))
        return rarity/len(self.models_)
