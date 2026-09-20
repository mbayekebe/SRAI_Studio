"""Model selection helpers."""
from __future__ import annotations
import numpy as np

def kfold_indices(n_samples,n_splits=5,seed=42,shuffle=True):
    idx=np.arange(n_samples)
    if shuffle:
        rng=np.random.default_rng(seed); rng.shuffle(idx)
    folds=np.array_split(idx,n_splits)
    for i in range(n_splits):
        test=folds[i]
        train=np.concatenate([folds[j] for j in range(n_splits) if j!=i])
        yield train,test

def cross_val_score(model_factory,X,y,metric,n_splits=5,seed=42):
    X=np.asarray(X); y=np.asarray(y)
    scores=[]
    for train_idx,test_idx in kfold_indices(len(X),n_splits,seed):
        model=model_factory()
        model.fit(X[train_idx],y[train_idx])
        pred=model.predict(X[test_idx])
        scores.append(metric(y[test_idx],pred))
    return np.asarray(scores,dtype=float)

def learning_curve(model_factory,X,y,metric,train_sizes,seed=42):
    rng=np.random.default_rng(seed)
    idx=np.arange(len(X)); rng.shuffle(idx)
    X=X[idx]; y=y[idx]
    rows=[]
    for n in train_sizes:
        n=int(n)
        split=max(1,int(.2*n))
        trainX=X[:n-split]; testX=X[n-split:n]
        trainy=y[:n-split]; testy=y[n-split:n]
        model=model_factory().fit(trainX,trainy)
        rows.append((n,metric(trainy,model.predict(trainX)),metric(testy,model.predict(testX))))
    return rows
