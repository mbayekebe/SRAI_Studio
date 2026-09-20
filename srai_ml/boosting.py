"""Simple boosting algorithms."""
from __future__ import annotations
import numpy as np
from .trees import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self,n_estimators=100,learning_rate=0.05,max_depth=2,seed=42):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.max_depth=max_depth; self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float)
        self.init_=float(np.mean(y))
        pred=np.full(len(y),self.init_)
        self.trees_=[]; self.train_loss_=[]
        for i in range(self.n_estimators):
            residual=y-pred
            tree=DecisionTreeRegressor(max_depth=self.max_depth,seed=self.seed+i).fit(X,residual)
            pred+=self.learning_rate*tree.predict(X)
            self.trees_.append(tree)
            self.train_loss_.append(float(np.mean((y-pred)**2)))
        return self
    def predict(self,X):
        pred=np.full(len(X),self.init_)
        for tree in self.trees_:
            pred+=self.learning_rate*tree.predict(X)
        return pred

class AdaBoostStumpClassifier:
    def __init__(self,n_estimators=50,learning_rate=1.0):
        self.n_estimators=n_estimators; self.learning_rate=learning_rate
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        y_signed=np.where(y==1,1,-1)
        n=len(y); w=np.full(n,1/n)
        self.models_=[]; self.alphas_=[]; self.errors_=[]
        for _ in range(self.n_estimators):
            best=None; best_error=np.inf
            for j in range(X.shape[1]):
                vals=np.unique(X[:,j]); thresholds=(vals[:-1]+vals[1:])/2
                for t in thresholds:
                    for polarity in (1,-1):
                        pred=np.ones(n,dtype=int)
                        pred[polarity*X[:,j] < polarity*t]=-1
                        err=np.sum(w[pred!=y_signed])
                        if err<best_error:
                            best_error=err; best=(j,float(t),polarity,pred.copy())
            if best is None or best_error>=0.5: break
            alpha=self.learning_rate*0.5*np.log((1-best_error)/(best_error+1e-12))
            j,t,p,pr=best
            w*=np.exp(-alpha*y_signed*pr); w/=w.sum()
            self.models_.append((j,t,p)); self.alphas_.append(alpha); self.errors_.append(float(best_error))
        return self
    def decision_function(self,X):
        X=np.asarray(X,float); score=np.zeros(len(X))
        for alpha,(j,t,p) in zip(self.alphas_,self.models_):
            pred=np.ones(len(X),dtype=int)
            pred[p*X[:,j] < p*t]=-1
            score+=alpha*pred
        return score
    def predict(self,X):
        return (self.decision_function(X)>=0).astype(int)
