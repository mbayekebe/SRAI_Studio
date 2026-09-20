"""Randomized algorithms and Monte Carlo utilities."""
from __future__ import annotations
import numpy as np

def monte_carlo_pi(samples=100000, seed=42):
    if samples <= 0:
        raise ValueError("samples must be positive.")
    rng=np.random.default_rng(seed)
    points=rng.uniform(-1,1,size=(samples,2))
    inside=np.sum(np.sum(points**2,axis=1)<=1)
    estimate=4*inside/samples
    se=4*np.sqrt((inside/samples)*(1-inside/samples)/samples)
    return float(estimate),float(se)

def importance_sampling_normal_tail(threshold,shift=3.0,samples=100000,seed=42):
    if samples<=0:
        raise ValueError("samples must be positive.")
    rng=np.random.default_rng(seed)
    x=rng.normal(shift,1,size=samples)
    weights=np.exp(-0.5*x**2+0.5*(x-shift)**2)
    estimate=np.mean((x>threshold)*weights)
    return float(estimate)

def random_projection_matrix(input_dim,output_dim,seed=42):
    if input_dim<=0 or output_dim<=0:
        raise ValueError("dimensions must be positive.")
    rng=np.random.default_rng(seed)
    return rng.normal(0,1/np.sqrt(output_dim),size=(input_dim,output_dim))

def random_project(X,output_dim,seed=42):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2:
        raise ValueError("X must be a matrix.")
    R=random_projection_matrix(X.shape[1],output_dim,seed)
    return X@R,R

def distance_distortion(X,Y):
    X=np.asarray(X,dtype=float); Y=np.asarray(Y,dtype=float)
    if X.shape[0]!=Y.shape[0]:
        raise ValueError("X and Y must have same row count.")
    def pairwise(Z):
        diff=Z[:,None,:]-Z[None,:,:]
        return np.sqrt(np.sum(diff**2,axis=2))
    DX=pairwise(X); DY=pairwise(Y)
    mask=DX>0
    ratios=DY[mask]/DX[mask]
    return {
        "mean_ratio":float(np.mean(ratios)),
        "max_abs_distortion":float(np.max(np.abs(ratios-1))),
        "median_abs_distortion":float(np.median(np.abs(ratios-1))),
    }

def count_sketch(X,output_dim,seed=42):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or output_dim<=0:
        raise ValueError("Invalid inputs.")
    rng=np.random.default_rng(seed)
    hashes=rng.integers(0,output_dim,size=X.shape[0])
    signs=rng.choice([-1.0,1.0],size=X.shape[0])
    S=np.zeros((output_dim,X.shape[0]))
    S[hashes,np.arange(X.shape[0])]=signs
    return S@X,S
