"""Activation functions and derivatives."""
from __future__ import annotations
import numpy as np

def sigmoid(x):
    x=np.clip(np.asarray(x,float),-500,500)
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s=sigmoid(x)
    return s*(1-s)

def tanh(x):
    return np.tanh(np.asarray(x,float))

def tanh_derivative(x):
    t=tanh(x)
    return 1-t*t

def relu(x):
    return np.maximum(np.asarray(x,float),0)

def relu_derivative(x):
    return (np.asarray(x,float)>0).astype(float)

def leaky_relu(x,alpha=0.01):
    x=np.asarray(x,float)
    return np.where(x>0,x,alpha*x)

def leaky_relu_derivative(x,alpha=0.01):
    x=np.asarray(x,float)
    return np.where(x>0,1.0,alpha)

def softmax(x,axis=-1):
    x=np.asarray(x,float)
    shifted=x-np.max(x,axis=axis,keepdims=True)
    e=np.exp(shifted)
    return e/np.sum(e,axis=axis,keepdims=True)
