"""Diffusion-model foundations."""
from __future__ import annotations
import numpy as np

def linear_beta_schedule(timesteps,beta_start=1e-4,beta_end=.02):
    return np.linspace(beta_start,beta_end,timesteps,dtype=float)

def diffusion_coefficients(betas):
    betas=np.asarray(betas,float); alphas=1-betas; alpha_bar=np.cumprod(alphas)
    return alphas,alpha_bar

def q_sample(x0,t,alpha_bar,noise=None,seed=42):
    x0=np.asarray(x0,float); ab=float(np.asarray(alpha_bar)[int(t)])
    if noise is None: noise=np.random.default_rng(seed).normal(size=x0.shape)
    noise=np.asarray(noise,float)
    xt=np.sqrt(ab)*x0+np.sqrt(1-ab)*noise
    return xt,noise

def predict_x0_from_noise(xt,t,alpha_bar,predicted_noise):
    ab=float(np.asarray(alpha_bar)[int(t)])
    return (np.asarray(xt,float)-np.sqrt(1-ab)*np.asarray(predicted_noise,float))/np.sqrt(ab)

def denoising_mse(true_noise,predicted_noise):
    return float(np.mean((np.asarray(true_noise,float)-np.asarray(predicted_noise,float))**2))

def sinusoidal_time_embedding(timesteps,dimension):
    t=np.asarray(timesteps,float)[:,None]; half=dimension//2
    frequencies=np.exp(-np.log(10000)*np.arange(half)/max(1,half-1))
    emb=np.concatenate([np.sin(t*frequencies),np.cos(t*frequencies)],axis=1)
    if dimension%2: emb=np.pad(emb,((0,0),(0,1)))
    return emb
