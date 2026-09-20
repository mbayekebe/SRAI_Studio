"""Small image-classification helpers."""
from __future__ import annotations
import numpy as np
from .convolution import conv2d_batch,max_pool2d,flatten_features
from .activations import relu,softmax

def edge_kernels():
    horizontal=np.array([[-1,-1,-1],[0,0,0],[1,1,1]],float)
    vertical=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],float)
    diagonal=np.array([[0,1,1],[-1,0,1],[-1,-1,0]],float)
    return np.stack([horizontal,vertical,diagonal])[:,None,:,:]

class SimpleCNNFeatureExtractor:
    def __init__(self,kernels=None,pool_size=2):
        self.kernels_=edge_kernels() if kernels is None else np.asarray(kernels,float)
        self.pool_size=pool_size
    def transform(self,images):
        X=np.asarray(images,float)
        features=relu(conv2d_batch(X,self.kernels_,padding=1))
        pooled=max_pool2d(features,self.pool_size)
        return flatten_features(pooled)

class SoftmaxClassifier:
    def __init__(self,learning_rate=0.1,max_iter=1000,l2=0.0):
        self.learning_rate=learning_rate; self.max_iter=max_iter; self.l2=l2
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        classes=np.unique(y)
        self.classes_=classes
        W=np.zeros((X.shape[1],len(classes))); b=np.zeros(len(classes))
        self.loss_history_=[]
        Y=np.eye(len(classes))[np.searchsorted(classes,y)]
        for _ in range(self.max_iter):
            P=softmax(X@W+b,axis=1)
            loss=-np.mean(np.sum(Y*np.log(np.clip(P,1e-12,1)),axis=1))
            loss+=.5*self.l2*np.sum(W*W)
            grad=(P-Y)/len(X)
            W-=self.learning_rate*(X.T@grad+self.l2*W)
            b-=self.learning_rate*np.sum(grad,axis=0)
            self.loss_history_.append(float(loss))
        self.W_=W; self.b_=b
        return self
    def predict_proba(self,X):
        return softmax(np.asarray(X,float)@self.W_+self.b_,axis=1)
    def predict(self,X):
        return self.classes_[np.argmax(self.predict_proba(X),axis=1)]
