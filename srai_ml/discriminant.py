"""Linear and quadratic discriminant analysis."""
from __future__ import annotations
import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self,regularization=1e-6):
        self.regularization=regularization
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        self.classes_=np.unique(y)
        self.means_=[]; self.priors_=[]
        covariance=np.zeros((X.shape[1],X.shape[1]))
        for c in self.classes_:
            Xc=X[y==c]
            mu=Xc.mean(axis=0)
            self.means_.append(mu)
            self.priors_.append(len(Xc)/len(X))
            centered=Xc-mu
            covariance+=centered.T@centered
        covariance/=max(1,len(X)-len(self.classes_))
        covariance+=self.regularization*np.eye(X.shape[1])
        self.inv_cov_=np.linalg.inv(covariance)
        self.means_=np.asarray(self.means_)
        self.priors_=np.asarray(self.priors_)
        return self
    def decision_function(self,X):
        X=np.asarray(X,float)
        scores=[]
        for mu,prior in zip(self.means_,self.priors_):
            scores.append(X@self.inv_cov_@mu-.5*mu@self.inv_cov_@mu+np.log(prior))
        return np.column_stack(scores)
    def predict(self,X):
        return self.classes_[np.argmax(self.decision_function(X),axis=1)]

class QuadraticDiscriminantAnalysis:
    def __init__(self,regularization=1e-6):
        self.regularization=regularization
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        self.classes_=np.unique(y)
        self.means_=[]; self.priors_=[]; self.inv_covs_=[]; self.logdets_=[]
        for c in self.classes_:
            Xc=X[y==c]
            mu=Xc.mean(axis=0)
            cov=np.cov(Xc,rowvar=False,ddof=1)+self.regularization*np.eye(X.shape[1])
            sign,logdet=np.linalg.slogdet(cov)
            self.means_.append(mu)
            self.priors_.append(len(Xc)/len(X))
            self.inv_covs_.append(np.linalg.inv(cov))
            self.logdets_.append(logdet)
        self.means_=np.asarray(self.means_); self.priors_=np.asarray(self.priors_)
        return self
    def decision_function(self,X):
        X=np.asarray(X,float); scores=[]
        for mu,prior,inv,logdet in zip(self.means_,self.priors_,self.inv_covs_,self.logdets_):
            d=X-mu
            quad=np.einsum("ij,jk,ik->i",d,inv,d)
            scores.append(-.5*(logdet+quad)+np.log(prior))
        return np.column_stack(scores)
    def predict(self,X):
        return self.classes_[np.argmax(self.decision_function(X),axis=1)]
