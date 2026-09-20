"""LSTM and GRU forward-pass cells."""
from __future__ import annotations
import numpy as np
from .activations import sigmoid

class LSTMCell:
    def __init__(self,input_dim,hidden_dim,seed=42):
        rng=np.random.default_rng(seed)
        scale=.2
        self.W=rng.normal(scale=scale,size=(input_dim+hidden_dim,4*hidden_dim))
        self.b=np.zeros(4*hidden_dim)
        self.hidden_dim=hidden_dim
    def step(self,x,h,c):
        combined=np.concatenate([x,h],axis=1)
        gates=combined@self.W+self.b
        i,f,g,o=np.split(gates,4,axis=1)
        i=sigmoid(i); f=sigmoid(f); o=sigmoid(o); g=np.tanh(g)
        c_new=f*c+i*g
        h_new=o*np.tanh(c_new)
        return h_new,c_new
    def forward(self,X,h0=None,c0=None):
        X=np.asarray(X,float)
        batch,time,_=X.shape
        h=np.zeros((batch,self.hidden_dim)) if h0 is None else np.asarray(h0,float)
        c=np.zeros_like(h) if c0 is None else np.asarray(c0,float)
        states=[]
        for t in range(time):
            h,c=self.step(X[:,t],h,c)
            states.append(h.copy())
        return np.stack(states,axis=1),(h,c)

class GRUCell:
    def __init__(self,input_dim,hidden_dim,seed=42):
        rng=np.random.default_rng(seed)
        scale=.2
        self.Wz=rng.normal(scale=scale,size=(input_dim+hidden_dim,hidden_dim))
        self.Wr=rng.normal(scale=scale,size=(input_dim+hidden_dim,hidden_dim))
        self.Wh=rng.normal(scale=scale,size=(input_dim+hidden_dim,hidden_dim))
        self.bz=np.zeros(hidden_dim); self.br=np.zeros(hidden_dim); self.bh=np.zeros(hidden_dim)
        self.hidden_dim=hidden_dim
    def step(self,x,h):
        combined=np.concatenate([x,h],axis=1)
        z=sigmoid(combined@self.Wz+self.bz)
        r=sigmoid(combined@self.Wr+self.br)
        candidate=np.tanh(np.concatenate([x,r*h],axis=1)@self.Wh+self.bh)
        h_new=(1-z)*h+z*candidate
        return h_new
    def forward(self,X,h0=None):
        X=np.asarray(X,float)
        batch,time,_=X.shape
        h=np.zeros((batch,self.hidden_dim)) if h0 is None else np.asarray(h0,float)
        states=[]
        for t in range(time):
            h=self.step(X[:,t],h)
            states.append(h.copy())
        return np.stack(states,axis=1),h
