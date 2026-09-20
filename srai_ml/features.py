"""Feature engineering utilities."""
from __future__ import annotations
import numpy as np

class PolynomialFeatures:
    def __init__(self,degree=2,include_bias=False):
        self.degree=degree
        self.include_bias=include_bias
    def fit(self,X):
        X=np.asarray(X,float)
        self.n_features_in_=X.shape[1]
        self.powers_=[]
        if self.include_bias:
            self.powers_.append(np.zeros(self.n_features_in_,dtype=int))
        def generate(total,start,prefix):
            if start==self.n_features_in_-1:
                self.powers_.append(np.array(prefix+[total],dtype=int))
                return
            for v in range(total+1):
                generate(total-v,start+1,prefix+[v])
        for d in range(1,self.degree+1):
            generate(d,0,[])
        return self
    def transform(self,X):
        X=np.asarray(X,float)
        cols=[]
        for power in self.powers_:
            cols.append(np.prod(X**power,axis=1))
        return np.column_stack(cols)
    def fit_transform(self,X):
        return self.fit(X).transform(X)

class OneHotEncoder:
    def fit(self,X):
        X=np.asarray(X,object)
        if X.ndim==1: X=X[:,None]
        self.categories_=[np.unique(X[:,j]) for j in range(X.shape[1])]
        return self
    def transform(self,X):
        X=np.asarray(X,object)
        if X.ndim==1: X=X[:,None]
        blocks=[]
        for j,cats in enumerate(self.categories_):
            blocks.append(np.column_stack([(X[:,j]==c).astype(float) for c in cats]))
        return np.column_stack(blocks)
    def fit_transform(self,X):
        return self.fit(X).transform(X)

def interaction_terms(X,pairs):
    X=np.asarray(X,float)
    return np.column_stack([X[:,i]*X[:,j] for i,j in pairs])

def log_transform(X,offset=1e-9):
    X=np.asarray(X,float)
    if np.any(X+offset<=0):
        raise ValueError("Log transform requires positive shifted values.")
    return np.log(X+offset)

def bin_numeric(x,bins):
    return np.digitize(np.asarray(x,float),np.asarray(bins,float),right=False)

def missing_indicator(X):
    X=np.asarray(X,float)
    return np.isnan(X).astype(float)

def median_impute(X):
    X=np.asarray(X,float).copy()
    med=np.nanmedian(X,axis=0)
    rows,cols=np.where(np.isnan(X))
    X[rows,cols]=med[cols]
    return X,med
