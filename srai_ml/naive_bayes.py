"""Naive Bayes classifiers."""
from __future__ import annotations
import numpy as np

class GaussianNB:
    def __init__(self,var_smoothing=1e-9):
        self.var_smoothing=var_smoothing
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        self.classes_=np.unique(y)
        self.means_=[]; self.vars_=[]; self.priors_=[]
        for c in self.classes_:
            Xc=X[y==c]
            self.means_.append(Xc.mean(axis=0))
            self.vars_.append(Xc.var(axis=0)+self.var_smoothing)
            self.priors_.append(len(Xc)/len(X))
        self.means_=np.asarray(self.means_)
        self.vars_=np.asarray(self.vars_)
        self.priors_=np.asarray(self.priors_)
        return self
    def _joint_log_likelihood(self,X):
        X=np.asarray(X,float)
        out=[]
        for i,c in enumerate(self.classes_):
            log_prior=np.log(self.priors_[i])
            log_density=-0.5*np.sum(np.log(2*np.pi*self.vars_[i])+
                                    (X-self.means_[i])**2/self.vars_[i],axis=1)
            out.append(log_prior+log_density)
        return np.column_stack(out)
    def predict_log_proba(self,X):
        jll=self._joint_log_likelihood(X)
        m=np.max(jll,axis=1,keepdims=True)
        log_norm=m+np.log(np.sum(np.exp(jll-m),axis=1,keepdims=True))
        return jll-log_norm
    def predict_proba(self,X):
        return np.exp(self.predict_log_proba(X))
    def predict(self,X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X),axis=1)]

class MultinomialNB:
    def __init__(self,alpha=1.0):
        self.alpha=alpha
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        if np.any(X<0): raise ValueError("MultinomialNB requires nonnegative features.")
        self.classes_=np.unique(y)
        self.class_log_prior_=[]
        self.feature_log_prob_=[]
        for c in self.classes_:
            Xc=X[y==c]
            count=Xc.sum(axis=0)+self.alpha
            self.feature_log_prob_.append(np.log(count/count.sum()))
            self.class_log_prior_.append(np.log(len(Xc)/len(X)))
        self.feature_log_prob_=np.asarray(self.feature_log_prob_)
        self.class_log_prior_=np.asarray(self.class_log_prior_)
        return self
    def predict(self,X):
        scores=np.asarray(X,float)@self.feature_log_prob_.T+self.class_log_prior_
        return self.classes_[np.argmax(scores,axis=1)]
