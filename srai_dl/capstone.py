"""Deep-learning capstone helpers."""
from __future__ import annotations
import numpy as np

def softmax(x):
    x=np.asarray(x,float)
    x=x-np.max(x,axis=1,keepdims=True)
    e=np.exp(x)
    return e/e.sum(axis=1,keepdims=True)

def train_linear_classifier(X,y,epochs=1000,lr=.1,l2=.001):
    X=np.asarray(X,float); y=np.asarray(y,int)
    classes=np.max(y)+1
    W=np.zeros((X.shape[1],classes)); b=np.zeros(classes)
    Y=np.eye(classes)[y]
    history=[]
    for _ in range(epochs):
        P=softmax(X@W+b)
        loss=-np.mean(np.sum(Y*np.log(np.clip(P,1e-12,1)),axis=1))+.5*l2*np.sum(W*W)
        grad=(P-Y)/len(X)
        W-=lr*(X.T@grad+l2*W)
        b-=lr*grad.sum(axis=0)
        history.append(float(loss))
    return W,b,history

def predict_linear_classifier(X,W,b):
    return np.argmax(np.asarray(X,float)@W+b,axis=1)

def reliability_summary(probabilities,predictions,labels):
    p=np.asarray(probabilities,float); pred=np.asarray(predictions,int); y=np.asarray(labels,int)
    return {
        "accuracy":float(np.mean(pred==y)),
        "mean_confidence":float(np.mean(np.max(p,axis=1))),
        "error_confidence":float(np.mean(np.max(p[pred!=y],axis=1))) if np.any(pred!=y) else 0.0,
    }
