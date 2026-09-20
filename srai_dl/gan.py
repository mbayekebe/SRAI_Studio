"""Generative-adversarial-network utilities."""
from __future__ import annotations
import numpy as np

def sigmoid(x):
    x=np.clip(np.asarray(x,float),-500,500); return 1/(1+np.exp(-x))

def discriminator_loss(real_logits,fake_logits):
    real=sigmoid(real_logits); fake=sigmoid(fake_logits)
    return float(-np.mean(np.log(real+1e-12)+np.log(1-fake+1e-12)))

def generator_loss(fake_logits,non_saturating=True):
    fake=sigmoid(fake_logits)
    if non_saturating: return float(-np.mean(np.log(fake+1e-12)))
    return float(np.mean(np.log(1-fake+1e-12)))

class LinearGenerator:
    def __init__(self,latent_dim,output_dim,seed=42):
        rng=np.random.default_rng(seed); self.W=rng.normal(scale=.5,size=(latent_dim,output_dim)); self.b=np.zeros(output_dim)
    def generate(self,z): return np.asarray(z,float)@self.W+self.b

class LinearDiscriminator:
    def __init__(self,input_dim,seed=42):
        rng=np.random.default_rng(seed); self.w=rng.normal(scale=.2,size=input_dim); self.b=0.0
    def logits(self,X): return np.asarray(X,float)@self.w+self.b
    def probabilities(self,X): return sigmoid(self.logits(X))

def mode_coverage(samples,centers):
    X=np.asarray(samples,float); C=np.asarray(centers,float)
    labels=np.argmin(((X[:,None,:]-C[None,:,:])**2).sum(axis=2),axis=1)
    counts=np.bincount(labels,minlength=len(C))
    return counts/len(X)
