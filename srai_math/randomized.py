import numpy as np
from scipy import stats
from srai_compat import dynamic_attribute
def monte_carlo_pi(samples,seed=42):
    r=np.random.default_rng(seed); inside=(r.random(samples)**2+r.random(samples)**2)<=1
    estimate=4*inside.mean(); return estimate,4*np.sqrt(inside.mean()*(1-inside.mean())/samples)
def importance_sampling_normal_tail(threshold,shift,samples,seed=42):
    r=np.random.default_rng(seed); x=r.normal(shift,1,samples); w=np.exp(-shift*x+shift**2/2)
    return float(np.mean((x>threshold)*w))
def random_project(x,n_components,seed=42):
    r=np.random.default_rng(seed); R=r.normal(size=(np.asarray(x).shape[1],n_components))/np.sqrt(n_components)
    return np.asarray(x)@R,R
def distance_distortion(x,y):
    from scipy.spatial.distance import pdist
    a=pdist(x); b=pdist(y); return {"mean":float(np.mean(np.abs(b/a-1))),"max":float(np.max(np.abs(b/a-1)))}
def count_sketch(x,n_components,seed=42):
    r=np.random.default_rng(seed); S=np.zeros((np.asarray(x).shape[1],n_components)); h=r.integers(0,n_components,len(S)); s=r.choice([-1,1],len(S)); S[np.arange(len(S)),h]=s
    return np.asarray(x)@S,S
__getattr__=dynamic_attribute
