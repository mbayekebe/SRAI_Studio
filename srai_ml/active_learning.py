"""Active-learning utilities."""
from __future__ import annotations
import numpy as np

def entropy_uncertainty(probabilities):
    p=np.clip(np.asarray(probabilities,float),1e-12,1)
    return -np.sum(p*np.log(p),axis=1)

def margin_uncertainty(probabilities):
    p=np.sort(np.asarray(probabilities,float),axis=1)
    return 1-(p[:,-1]-p[:,-2])

def least_confidence(probabilities):
    p=np.asarray(probabilities,float)
    return 1-np.max(p,axis=1)

def query_top_k(probabilities,k=1,strategy="entropy"):
    if strategy=="entropy":
        scores=entropy_uncertainty(probabilities)
    elif strategy=="margin":
        scores=margin_uncertainty(probabilities)
    elif strategy=="least_confidence":
        scores=least_confidence(probabilities)
    else:
        raise ValueError("Unsupported strategy.")
    return np.argsort(scores)[::-1][:k],scores

def active_learning_round(model_factory,X_pool,y_pool,labeled_indices,query_size=10,strategy="entropy"):
    X=np.asarray(X_pool,float); y=np.asarray(y_pool,int)
    labeled=np.asarray(labeled_indices,int)
    model=model_factory().fit(X[labeled],y[labeled])
    unlabeled=np.setdiff1d(np.arange(len(X)),labeled)
    proba=model.predict_proba(X[unlabeled])
    chosen_local,_=query_top_k(proba,query_size,strategy)
    chosen=unlabeled[chosen_local]
    return model,chosen
