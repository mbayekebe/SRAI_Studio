import numpy as np
def random_mask(X,rate=.2,seed=42):
    rng=np.random.default_rng(seed); m=rng.random(np.asarray(X).shape)>=rate
    return np.asarray(X)*m,m
def contrastive_pairs(X,noise=.1,seed=42):
    rng=np.random.default_rng(seed); X=np.asarray(X,float)
    return X+rng.normal(scale=noise,size=X.shape),X+rng.normal(scale=noise,size=X.shape)
def cosine_matrix(A,B):
    A=np.asarray(A,float); B=np.asarray(B,float)
    A=A/np.clip(np.linalg.norm(A,axis=1,keepdims=True),1e-12,None)
    B=B/np.clip(np.linalg.norm(B,axis=1,keepdims=True),1e-12,None)
    return A@B.T
def nt_xent_loss(Z1,Z2,temp=.1):
    S=cosine_matrix(Z1,Z2)/temp; e=np.exp(S-S.max(1,keepdims=True)); p=e/e.sum(1,keepdims=True)
    return float(-np.mean(np.log(np.clip(np.diag(p),1e-12,1))))
