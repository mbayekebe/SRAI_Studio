"""Sequential neural network implemented with NumPy."""
from __future__ import annotations
import numpy as np
from .layers import Dropout
from .activations import sigmoid,softmax
from .losses import binary_cross_entropy,mse_loss,categorical_cross_entropy

class Sequential:
    def __init__(self,layers):
        self.layers=list(layers)
    def forward(self,X,training=True):
        out=np.asarray(X,float)
        for layer in self.layers:
            if isinstance(layer,Dropout):
                out=layer.forward(out,training=training)
            else:
                out=layer.forward(out)
        return out
    def backward(self,grad):
        out=np.asarray(grad,float)
        for layer in self.layers[::-1]:
            out=layer.backward(out)
        return out
    def parameters_and_gradients(self):
        for layer in self.layers:
            for p,g in zip(layer.parameters(),layer.gradients()):
                yield p,g
    def predict(self,X):
        return self.forward(X,training=False)

def binary_output_gradient(y_true,y_prob):
    y=np.asarray(y_true,float).reshape(-1,1)
    p=np.asarray(y_prob,float).reshape(-1,1)
    return (p-y)/len(y)

def mse_output_gradient(y_true,y_pred):
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float)
    return 2*(p-y)/y.size

def categorical_output_gradient(y_true,y_prob):
    y=np.asarray(y_true)
    p=np.asarray(y_prob,float).copy()
    if y.ndim==1:
        p[np.arange(len(y)),y.astype(int)]-=1
    else:
        p-=y
    return p/len(p)
