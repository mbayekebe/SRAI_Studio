"""Autoencoder utilities."""
from __future__ import annotations
import numpy as np

class LinearAutoencoder:
    def __init__(self,latent_dim): self.latent_dim=latent_dim
    def fit(self,X):
        X=np.asarray(X,float); self.mean_=X.mean(axis=0); Xc=X-self.mean_
        _,_,Vt=np.linalg.svd(Xc,full_matrices=False)
        self.components_=Vt[:self.latent_dim]
        return self
    def encode(self,X): return (np.asarray(X,float)-self.mean_)@self.components_.T
    def decode(self,Z): return np.asarray(Z,float)@self.components_+self.mean_
    def reconstruct(self,X): return self.decode(self.encode(X))

class DenoisingAutoencoder:
    def __init__(self,latent_dim,noise_std=.1,seed=42):
        self.latent_dim=latent_dim; self.noise_std=noise_std; self.seed=seed
    def fit(self,X):
        X=np.asarray(X,float); rng=np.random.default_rng(self.seed)
        noisy=X+rng.normal(scale=self.noise_std,size=X.shape)
        self.mean_noisy_=noisy.mean(axis=0); self.mean_clean_=X.mean(axis=0)
        A=noisy-self.mean_noisy_; B=X-self.mean_clean_
        U,s,Vt=np.linalg.svd(A,full_matrices=False)
        Z=U[:,:self.latent_dim]*s[:self.latent_dim]
        self.encoder_=Vt[:self.latent_dim]
        self.decoder_=np.linalg.pinv(Z)@B
        return self
    def encode(self,X): return (np.asarray(X,float)-self.mean_noisy_)@self.encoder_.T
    def decode(self,Z): return np.asarray(Z,float)@self.decoder_+self.mean_clean_
    def reconstruct(self,X): return self.decode(self.encode(X))

def reconstruction_error(X,reconstructed):
    X=np.asarray(X,float); R=np.asarray(reconstructed,float)
    return float(np.mean((X-R)**2))

def anomaly_scores(X,reconstructed):
    X=np.asarray(X,float); R=np.asarray(reconstructed,float)
    return np.mean((X-R)**2,axis=1)
