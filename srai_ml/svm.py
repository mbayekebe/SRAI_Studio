"""Linear support vector classifier using subgradient descent."""
from __future__ import annotations
import numpy as np

class LinearSVM:
    def __init__(self,learning_rate=0.01,lambda_reg=0.01,max_iter=1000):
        self.learning_rate=learning_rate; self.lambda_reg=lambda_reg; self.max_iter=max_iter
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.where(np.asarray(y)==1,1,-1)
        w=np.zeros(X.shape[1]); b=0.0; self.history_=[]
        for _ in range(self.max_iter):
            margins=y*(X@w+b)
            mask=margins<1
            grad_w=self.lambda_reg*w-(X[mask].T@y[mask])/len(y)
            grad_b=-np.sum(y[mask])/len(y)
            w-=self.learning_rate*grad_w; b-=self.learning_rate*grad_b
            hinge=np.maximum(0,1-margins)
            self.history_.append(float(.5*self.lambda_reg*w@w+np.mean(hinge)))
        self.coef_=w; self.intercept_=float(b)
        return self
    def decision_function(self,X):
        return np.asarray(X,float)@self.coef_+self.intercept_
    def predict(self,X):
        return (self.decision_function(X)>=0).astype(int)
