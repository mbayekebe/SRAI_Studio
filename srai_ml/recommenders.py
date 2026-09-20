"""Recommender-system utilities."""
from __future__ import annotations
import numpy as np

def cosine_similarity_matrix(X):
    X=np.asarray(X,float)
    norms=np.linalg.norm(X,axis=1,keepdims=True)
    norms[norms==0]=1.0
    Z=X/norms
    return Z@Z.T

class UserKNNRecommender:
    def __init__(self,n_neighbors=5):
        self.n_neighbors=n_neighbors
    def fit(self,ratings):
        self.ratings_=np.asarray(ratings,float)
        self.similarity_=cosine_similarity_matrix(np.nan_to_num(self.ratings_,nan=0.0))
        return self
    def predict(self,user,item):
        sims=self.similarity_[user].copy()
        sims[user]=-np.inf
        candidates=np.where(~np.isnan(self.ratings_[:,item]))[0]
        if candidates.size==0:
            return float(np.nanmean(self.ratings_))
        order=candidates[np.argsort(sims[candidates])[::-1][:self.n_neighbors]]
        weights=np.maximum(sims[order],0)
        values=self.ratings_[order,item]
        if np.sum(weights)<=1e-12:
            return float(np.nanmean(values))
        return float(np.sum(weights*values)/np.sum(weights))
    def recommend(self,user,n=5):
        unseen=np.where(np.isnan(self.ratings_[user]))[0]
        scores=[(item,self.predict(user,item)) for item in unseen]
        return sorted(scores,key=lambda x:x[1],reverse=True)[:n]

class MatrixFactorization:
    def __init__(self,n_factors=10,learning_rate=0.01,regularization=0.02,epochs=100,seed=42):
        self.n_factors=n_factors; self.learning_rate=learning_rate
        self.regularization=regularization; self.epochs=epochs; self.seed=seed
    def fit(self,ratings):
        R=np.asarray(ratings,float)
        rng=np.random.default_rng(self.seed)
        self.user_factors_=rng.normal(scale=.1,size=(R.shape[0],self.n_factors))
        self.item_factors_=rng.normal(scale=.1,size=(R.shape[1],self.n_factors))
        self.global_mean_=float(np.nanmean(R))
        observed=np.argwhere(~np.isnan(R))
        self.loss_history_=[]
        for _ in range(self.epochs):
            rng.shuffle(observed)
            sq=0.0
            for u,i in observed:
                pred=self.global_mean_+self.user_factors_[u]@self.item_factors_[i]
                err=R[u,i]-pred
                pu=self.user_factors_[u].copy()
                qi=self.item_factors_[i].copy()
                self.user_factors_[u]+=self.learning_rate*(err*qi-self.regularization*pu)
                self.item_factors_[i]+=self.learning_rate*(err*pu-self.regularization*qi)
                sq+=err**2
            self.loss_history_.append(float(sq/len(observed)))
        return self
    def predict(self,user,item):
        return float(self.global_mean_+self.user_factors_[user]@self.item_factors_[item])
    def recommend(self,user,ratings,n=5):
        R=np.asarray(ratings,float)
        unseen=np.where(np.isnan(R[user]))[0]
        scores=[(i,self.predict(user,i)) for i in unseen]
        return sorted(scores,key=lambda x:x[1],reverse=True)[:n]

def precision_at_k(recommended,relevant,k):
    rec=[item for item,_ in recommended[:k]]
    rel=set(relevant)
    return float(sum(item in rel for item in rec)/k)

def recall_at_k(recommended,relevant,k):
    rec={item for item,_ in recommended[:k]}
    rel=set(relevant)
    return float(len(rec&rel)/len(rel)) if rel else 0.0
