"""Linear and logistic models implemented with NumPy."""
from __future__ import annotations
import numpy as np
from .preprocessing import add_intercept

class LinearRegression:
    def __init__(self,fit_intercept=True):
        self.fit_intercept=fit_intercept
    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
        A=add_intercept(X) if self.fit_intercept else X
        self.coef_full_=np.linalg.pinv(A)@y
        if self.fit_intercept:
            self.intercept_=float(self.coef_full_[0]); self.coef_=self.coef_full_[1:]
        else:
            self.intercept_=0.0; self.coef_=self.coef_full_
        return self
    def predict(self,X):
        return np.asarray(X,dtype=float)@self.coef_+self.intercept_

class RidgeRegression:
    def __init__(self,alpha=1.0,fit_intercept=True):
        self.alpha=float(alpha); self.fit_intercept=fit_intercept
    def fit(self,X,y):
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
        A=add_intercept(X) if self.fit_intercept else X
        penalty=np.eye(A.shape[1])*self.alpha
        if self.fit_intercept: penalty[0,0]=0
        w=np.linalg.solve(A.T@A+penalty,A.T@y)
        if self.fit_intercept:
            self.intercept_=float(w[0]); self.coef_=w[1:]
        else:
            self.intercept_=0.0; self.coef_=w
        return self
    def predict(self,X):
        return np.asarray(X,dtype=float)@self.coef_+self.intercept_

class LogisticRegressionGD:
    def __init__(self,learning_rate=0.1,max_iter=1000,l2=0.0):
        self.learning_rate=learning_rate; self.max_iter=max_iter; self.l2=l2
    @staticmethod
    def _sigmoid(z):
        z=np.clip(z,-500,500)
        return 1/(1+np.exp(-z))
    def fit(self,X,y):
        X=add_intercept(X); y=np.asarray(y,dtype=float)
        w=np.zeros(X.shape[1]); self.history_=[]
        for _ in range(self.max_iter):
            p=self._sigmoid(X@w)
            eps=1e-12
            loss=-np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))
            loss+=0.5*self.l2*np.sum(w[1:]**2)
            self.history_.append(float(loss))
            grad=X.T@(p-y)/len(y)
            grad[1:]+=self.l2*w[1:]
            w-=self.learning_rate*grad
        self.intercept_=float(w[0]); self.coef_=w[1:]
        return self
    def predict_proba(self,X):
        p=self._sigmoid(np.asarray(X,dtype=float)@self.coef_+self.intercept_)
        return np.column_stack([1-p,p])
    def predict(self,X,threshold=0.5):
        return (self.predict_proba(X)[:,1]>=threshold).astype(int)
