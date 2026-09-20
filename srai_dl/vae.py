"""Variational autoencoder mathematics."""
from __future__ import annotations
import numpy as np

def reparameterize(mean,log_variance,seed=42):
    mean=np.asarray(mean,float); logv=np.asarray(log_variance,float)
    eps=np.random.default_rng(seed).normal(size=mean.shape)
    return mean+np.exp(.5*logv)*eps

def kl_standard_normal(mean,log_variance):
    mean=np.asarray(mean,float); logv=np.asarray(log_variance,float)
    return float(-.5*np.mean(np.sum(1+logv-mean**2-np.exp(logv),axis=-1)))

def binary_reconstruction_loss(X,reconstruction):
    X=np.asarray(X,float); R=np.clip(np.asarray(reconstruction,float),1e-12,1-1e-12)
    return float(-np.mean(np.sum(X*np.log(R)+(1-X)*np.log(1-R),axis=-1)))

def elbo_loss(X,reconstruction,mean,log_variance,beta=1.0):
    rec=binary_reconstruction_loss(X,reconstruction)
    kl=kl_standard_normal(mean,log_variance)
    return float(rec+beta*kl),rec,kl

def gaussian_decoder(z,W,b):
    return np.asarray(z,float)@np.asarray(W,float)+np.asarray(b,float)

def latent_interpolation(z0,z1,steps=10):
    z0=np.asarray(z0,float); z1=np.asarray(z1,float)
    a=np.linspace(0,1,steps)[:,None]
    return (1-a)*z0+a*z1
