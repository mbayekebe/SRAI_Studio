"""Basic neural-network layers."""
from __future__ import annotations
import numpy as np
from .activations import relu,relu_derivative,sigmoid,sigmoid_derivative,tanh,tanh_derivative

class Dense:
    def __init__(self,input_dim,output_dim,seed=42,scale=None):
        rng=np.random.default_rng(seed)
        if scale is None:
            scale=np.sqrt(2/input_dim)
        self.W=rng.normal(0,scale,size=(input_dim,output_dim))
        self.b=np.zeros(output_dim)
    def forward(self,X):
        self.X=np.asarray(X,float)
        self.Z=self.X@self.W+self.b
        return self.Z
    def backward(self,grad_output):
        grad=np.asarray(grad_output,float)
        self.grad_W=self.X.T@grad/len(self.X)
        self.grad_b=grad.mean(axis=0)
        return grad@self.W.T
    def parameters(self):
        return [self.W,self.b]
    def gradients(self):
        return [self.grad_W,self.grad_b]

class Activation:
    def __init__(self,name):
        self.name=name
    def forward(self,X):
        self.X=np.asarray(X,float)
        if self.name=="relu": return relu(self.X)
        if self.name=="sigmoid": return sigmoid(self.X)
        if self.name=="tanh": return tanh(self.X)
        if self.name=="linear": return self.X
        raise ValueError("Unsupported activation.")
    def backward(self,grad_output):
        if self.name=="relu": d=relu_derivative(self.X)
        elif self.name=="sigmoid": d=sigmoid_derivative(self.X)
        elif self.name=="tanh": d=tanh_derivative(self.X)
        elif self.name=="linear": d=np.ones_like(self.X)
        else: raise ValueError("Unsupported activation.")
        return np.asarray(grad_output,float)*d
    def parameters(self): return []
    def gradients(self): return []

class Dropout:
    def __init__(self,rate=0.5,seed=42):
        if not 0<=rate<1: raise ValueError("rate must lie in [0,1).")
        self.rate=rate
        self.rng=np.random.default_rng(seed)
    def forward(self,X,training=True):
        X=np.asarray(X,float)
        if not training or self.rate==0:
            self.mask=np.ones_like(X)
            return X
        self.mask=(self.rng.random(X.shape)>=self.rate)/(1-self.rate)
        return X*self.mask
    def backward(self,grad_output):
        return np.asarray(grad_output,float)*self.mask
    def parameters(self): return []
    def gradients(self): return []
