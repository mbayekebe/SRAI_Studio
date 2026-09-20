"""Training utilities for NumPy neural networks."""
from __future__ import annotations
import numpy as np
from .losses import binary_cross_entropy,mse_loss
from .network import binary_output_gradient,mse_output_gradient

def train_binary(network,optimizer,X,y,epochs=500,l2=0.0):
    y=np.asarray(y,float).reshape(-1,1)
    history=[]
    for _ in range(epochs):
        pred=network.forward(X,training=True)
        loss=binary_cross_entropy(y,pred)
        if l2:
            penalty=0.0
            for layer in network.layers:
                if hasattr(layer,"W"):
                    penalty+=np.sum(layer.W**2)
            loss+=0.5*l2*penalty
        grad=binary_output_gradient(y,pred)
        network.backward(grad)
        pairs=[]
        for layer in network.layers:
            if hasattr(layer,"W") and l2:
                layer.grad_W+=l2*layer.W
            for p,g in zip(layer.parameters(),layer.gradients()):
                pairs.append((p,g))
        optimizer.step(pairs)
        history.append(float(loss))
    return history

def train_regression(network,optimizer,X,y,epochs=500,l2=0.0):
    y=np.asarray(y,float)
    if y.ndim==1: y=y[:,None]
    history=[]
    for _ in range(epochs):
        pred=network.forward(X,training=True)
        loss=mse_loss(y,pred)
        grad=mse_output_gradient(y,pred)
        network.backward(grad)
        pairs=[]
        for layer in network.layers:
            if hasattr(layer,"W") and l2:
                layer.grad_W+=l2*layer.W
            for p,g in zip(layer.parameters(),layer.gradients()):
                pairs.append((p,g))
        optimizer.step(pairs)
        history.append(float(loss))
    return history

def early_stopping(history,patience=10,min_delta=0.0):
    best=np.inf; wait=0
    for i,value in enumerate(history):
        if value<best-min_delta:
            best=value; wait=0
        else:
            wait+=1
            if wait>=patience:
                return i
    return len(history)-1

def l2_penalty(weights):
    return float(sum(np.sum(np.asarray(w,float)**2) for w in weights))
