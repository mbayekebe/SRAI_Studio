"""Representation-learning utilities."""
from __future__ import annotations
import numpy as np

def normalize_embeddings(Z):
    Z=np.asarray(Z,float); n=np.linalg.norm(Z,axis=1,keepdims=True); n[n==0]=1
    return Z/n

def cosine_similarity_matrix(Z):
    Z=normalize_embeddings(Z); return Z@Z.T

def contrastive_loss(anchor,positive,negative,margin=1.0):
    a=np.asarray(anchor,float); p=np.asarray(positive,float); n=np.asarray(negative,float)
    dpos=np.linalg.norm(a-p,axis=-1); dneg=np.linalg.norm(a-n,axis=-1)
    return float(np.mean(np.maximum(0,dpos-dneg+margin)))

def nt_xent_loss(z1,z2,temperature=.1):
    z1=normalize_embeddings(z1); z2=normalize_embeddings(z2)
    logits=z1@z2.T/temperature
    logits=logits-np.max(logits,axis=1,keepdims=True)
    probs=np.exp(logits); probs/=probs.sum(axis=1,keepdims=True)
    return float(-np.mean(np.log(np.clip(np.diag(probs),1e-12,1))))

def linear_probe_fit(Z,y,l2=1e-6):
    Z=np.asarray(Z,float); y=np.asarray(y,float)
    A=np.column_stack([np.ones(len(Z)),Z]); I=np.eye(A.shape[1]); I[0,0]=0
    return np.linalg.solve(A.T@A+l2*I,A.T@y)

def linear_probe_predict(Z,weights):
    Z=np.asarray(Z,float); w=np.asarray(weights,float)
    return np.column_stack([np.ones(len(Z)),Z])@w

def retrieval_precision_at_k(embeddings,labels,k=5):
    Z=normalize_embeddings(embeddings); labels=np.asarray(labels)
    S=Z@Z.T; np.fill_diagonal(S,-np.inf)
    scores=[]
    for i in range(len(Z)):
        idx=np.argsort(S[i])[::-1][:k]; scores.append(np.mean(labels[idx]==labels[i]))
    return float(np.mean(scores))
