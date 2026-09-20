"""Fine-tuning foundations."""
from __future__ import annotations
import numpy as np

def cross_entropy_loss(logits,targets):
    x=np.asarray(logits,float)
    y=np.asarray(targets,int)
    x=x-x.max(axis=1,keepdims=True)
    p=np.exp(x); p/=p.sum(axis=1,keepdims=True)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))))

def train_linear_head(features,targets,classes=None,learning_rate=.1,epochs=1000,l2=.0):
    X=np.asarray(features,float)
    y=np.asarray(targets,int)
    n_classes=int(np.max(y)+1 if classes is None else classes)
    W=np.zeros((X.shape[1],n_classes)); b=np.zeros(n_classes)
    Y=np.eye(n_classes)[y]
    history=[]
    for _ in range(epochs):
        logits=X@W+b
        shifted=logits-logits.max(axis=1,keepdims=True)
        p=np.exp(shifted); p/=p.sum(axis=1,keepdims=True)
        loss=-np.mean(np.sum(Y*np.log(np.clip(p,1e-12,1)),axis=1))+.5*l2*np.sum(W*W)
        grad=(p-Y)/len(X)
        W-=learning_rate*(X.T@grad+l2*W)
        b-=learning_rate*grad.sum(axis=0)
        history.append(float(loss))
    return W,b,history

def catastrophic_forgetting_score(before_accuracy,after_accuracy):
    return float(max(0.0,before_accuracy-after_accuracy))

def learning_rate_schedule(step,warmup_steps,total_steps,peak_lr):
    if step<warmup_steps:
        return float(peak_lr*(step+1)/max(warmup_steps,1))
    progress=(step-warmup_steps)/max(total_steps-warmup_steps,1)
    return float(peak_lr*max(0.0,1-progress))
