import numpy as np
class FrozenLinearBackbone:
    def __init__(self,input_dim,feature_dim,seed=42):
        self.W=np.random.default_rng(seed).normal(scale=.2,size=(input_dim,feature_dim))
    def transform(self,X): return np.asarray(X,float)@self.W
class LinearHead:
    def __init__(self,lr=.1,iters=1000): self.lr=lr; self.iters=iters
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int); k=np.max(y)+1
        self.W=np.zeros((X.shape[1],k)); self.b=np.zeros(k); Y=np.eye(k)[y]
        for _ in range(self.iters):
            z=X@self.W+self.b; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
            g=(p-Y)/len(X); self.W-=self.lr*X.T@g; self.b-=self.lr*g.sum(0)
        return self
    def predict(self,X): return np.argmax(np.asarray(X)@self.W+self.b,axis=1)
def fine_tune_step(backbone,head,X,y,lr=.01):
    F=backbone.transform(X); pred=head.predict(F); return float(np.mean(pred==y))
