"""Simple decision trees and random forests."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def gini_impurity(y):
    y=np.asarray(y,dtype=int)
    if y.size==0: return 0.0
    counts=np.bincount(y)
    p=counts[counts>0]/y.size
    return float(1-np.sum(p*p))

def variance_impurity(y):
    y=np.asarray(y,dtype=float)
    return float(np.var(y)) if y.size else 0.0

@dataclass
class Node:
    feature:int|None=None
    threshold:float|None=None
    left:object|None=None
    right:object|None=None
    value:float|int|None=None

class DecisionTreeClassifier:
    def __init__(self,max_depth=5,min_samples_split=2,max_features=None,seed=42):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split
        self.max_features=max_features; self.seed=seed
    def fit(self,X,y):
        self.rng_=np.random.default_rng(self.seed)
        self.n_classes_=int(np.max(y))+1
        self.root_=self._grow(np.asarray(X,float),np.asarray(y,int),0)
        return self
    def _best_split(self,X,y):
        n_features=X.shape[1]
        features=np.arange(n_features)
        if self.max_features is not None:
            k=min(n_features,int(self.max_features))
            features=self.rng_.choice(features,size=k,replace=False)
        best_gain=-1; best=None
        parent=gini_impurity(y)
        for j in features:
            values=np.unique(X[:,j])
            thresholds=(values[:-1]+values[1:])/2
            for t in thresholds:
                left=X[:,j]<=t
                if left.sum()==0 or left.sum()==len(y): continue
                impurity=(left.mean()*gini_impurity(y[left])+
                          (~left).mean()*gini_impurity(y[~left]))
                gain=parent-impurity
                if gain>best_gain:
                    best_gain=gain; best=(j,float(t),left)
        return best
    def _grow(self,X,y,depth):
        counts=np.bincount(y,minlength=self.n_classes_)
        value=int(np.argmax(counts))
        if depth>=self.max_depth or len(y)<self.min_samples_split or len(np.unique(y))==1:
            return Node(value=value)
        split=self._best_split(X,y)
        if split is None: return Node(value=value)
        j,t,left=split
        return Node(j,t,self._grow(X[left],y[left],depth+1),
                    self._grow(X[~left],y[~left],depth+1),value)
    def _predict_one(self,x,node):
        if node.feature is None: return node.value
        return self._predict_one(x,node.left if x[node.feature]<=node.threshold else node.right)
    def predict(self,X):
        X=np.asarray(X,float)
        return np.array([self._predict_one(x,self.root_) for x in X],dtype=int)

class DecisionTreeRegressor:
    def __init__(self,max_depth=5,min_samples_split=2,max_features=None,seed=42):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split
        self.max_features=max_features; self.seed=seed
    def fit(self,X,y):
        self.rng_=np.random.default_rng(self.seed)
        self.root_=self._grow(np.asarray(X,float),np.asarray(y,float),0)
        return self
    def _best_split(self,X,y):
        features=np.arange(X.shape[1])
        if self.max_features is not None:
            features=self.rng_.choice(features,size=min(len(features),int(self.max_features)),replace=False)
        parent=variance_impurity(y); best_gain=-1; best=None
        for j in features:
            vals=np.unique(X[:,j]); thresholds=(vals[:-1]+vals[1:])/2
            for t in thresholds:
                left=X[:,j]<=t
                if left.sum()==0 or left.sum()==len(y): continue
                impurity=left.mean()*variance_impurity(y[left])+(~left).mean()*variance_impurity(y[~left])
                gain=parent-impurity
                if gain>best_gain:
                    best_gain=gain; best=(j,float(t),left)
        return best
    def _grow(self,X,y,depth):
        value=float(np.mean(y))
        if depth>=self.max_depth or len(y)<self.min_samples_split or np.isclose(np.var(y),0):
            return Node(value=value)
        split=self._best_split(X,y)
        if split is None: return Node(value=value)
        j,t,left=split
        return Node(j,t,self._grow(X[left],y[left],depth+1),
                    self._grow(X[~left],y[~left],depth+1),value)
    def _predict_one(self,x,node):
        if node.feature is None: return node.value
        return self._predict_one(x,node.left if x[node.feature]<=node.threshold else node.right)
    def predict(self,X):
        return np.array([self._predict_one(x,self.root_) for x in np.asarray(X,float)])

class RandomForestClassifier:
    def __init__(self,n_estimators=50,max_depth=5,max_features=None,seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.max_features=max_features; self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        rng=np.random.default_rng(self.seed)
        self.trees_=[]
        mf=self.max_features or max(1,int(np.sqrt(X.shape[1])))
        for i in range(self.n_estimators):
            idx=rng.integers(0,len(X),len(X))
            tree=DecisionTreeClassifier(self.max_depth,max_features=mf,seed=self.seed+i)
            tree.fit(X[idx],y[idx]); self.trees_.append(tree)
        return self
    def predict(self,X):
        votes=np.array([t.predict(X) for t in self.trees_])
        return np.apply_along_axis(lambda c:np.bincount(c).argmax(),0,votes)

class RandomForestRegressor:
    def __init__(self,n_estimators=50,max_depth=5,max_features=None,seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.max_features=max_features; self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float)
        rng=np.random.default_rng(self.seed)
        self.trees_=[]
        mf=self.max_features or max(1,int(np.sqrt(X.shape[1])))
        for i in range(self.n_estimators):
            idx=rng.integers(0,len(X),len(X))
            tree=DecisionTreeRegressor(self.max_depth,max_features=mf,seed=self.seed+i)
            tree.fit(X[idx],y[idx]); self.trees_.append(tree)
        return self
    def predict(self,X):
        return np.mean([t.predict(X) for t in self.trees_],axis=0)
