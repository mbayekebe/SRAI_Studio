"""Model-agnostic interpretability utilities."""
from __future__ import annotations
import numpy as np

def permutation_importance(model,X,y,metric,n_repeats=5,seed=42,higher_is_better=True):
    X=np.asarray(X,float); y=np.asarray(y)
    rng=np.random.default_rng(seed)
    baseline=metric(y,model.predict(X))
    importance=np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        values=[]
        for _ in range(n_repeats):
            Xp=X.copy()
            rng.shuffle(Xp[:,j])
            score=metric(y,model.predict(Xp))
            values.append((baseline-score) if higher_is_better else (score-baseline))
        importance[j]=np.mean(values)
    return importance

def partial_dependence(model,X,feature,grid):
    X=np.asarray(X,float)
    values=[]
    for v in grid:
        Xp=X.copy()
        Xp[:,feature]=v
        values.append(float(np.mean(model.predict(Xp))))
    return np.asarray(values)

def local_linear_explanation(model,x,scale=0.1,samples=500,seed=42):
    x=np.asarray(x,float)
    rng=np.random.default_rng(seed)
    Z=x+rng.normal(scale=scale,size=(samples,x.size))
    y=model.predict(Z)
    A=np.column_stack([np.ones(samples),Z-x])
    coef=np.linalg.pinv(A)@y
    return {"intercept":float(coef[0]),"local_coefficients":coef[1:]}

def feature_effect_correlation(model,X,feature):
    X=np.asarray(X,float)
    preds=np.asarray(model.predict(X),float)
    return float(np.corrcoef(X[:,feature],preds)[0,1])
