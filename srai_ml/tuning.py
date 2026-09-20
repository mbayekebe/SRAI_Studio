"""Hyperparameter search utilities."""
from __future__ import annotations
import itertools
import numpy as np

def parameter_grid(grid):
    keys=list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys,values))

def grid_search(model_factory,param_grid,X,y,metric,cv_splitter):
    results=[]
    for params in parameter_grid(param_grid):
        scores=[]
        for train_idx,val_idx in cv_splitter:
            model=model_factory(**params).fit(X[train_idx],y[train_idx])
            scores.append(metric(y[val_idx],model.predict(X[val_idx])))
        results.append({"params":params,"mean_score":float(np.mean(scores)),
                        "std_score":float(np.std(scores,ddof=1) if len(scores)>1 else 0.0)})
    return results

def random_search(model_factory,param_distributions,n_iter,X,y,metric,cv_splitter,seed=42):
    rng=np.random.default_rng(seed)
    keys=list(param_distributions)
    results=[]
    for _ in range(n_iter):
        params={k:rng.choice(param_distributions[k]).item() if hasattr(rng.choice(param_distributions[k]),"item")
                else rng.choice(param_distributions[k]) for k in keys}
        scores=[]
        for train_idx,val_idx in cv_splitter:
            model=model_factory(**params).fit(X[train_idx],y[train_idx])
            scores.append(metric(y[val_idx],model.predict(X[val_idx])))
        results.append({"params":params,"mean_score":float(np.mean(scores))})
    return results

def select_best(results,higher_is_better=True):
    key=lambda row:row["mean_score"]
    return max(results,key=key) if higher_is_better else min(results,key=key)
