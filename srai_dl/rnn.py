"""Recurrent neural-network utilities."""
from __future__ import annotations
import numpy as np
from .activations import tanh,softmax

class SimpleRNN:
    def __init__(self,input_dim,hidden_dim,output_dim,seed=42):
        rng=np.random.default_rng(seed)
        self.Wx=rng.normal(scale=.2,size=(input_dim,hidden_dim))
        self.Wh=rng.normal(scale=.2,size=(hidden_dim,hidden_dim))
        self.bh=np.zeros(hidden_dim)
        self.Wy=rng.normal(scale=.2,size=(hidden_dim,output_dim))
        self.by=np.zeros(output_dim)
    def forward(self,X,h0=None):
        X=np.asarray(X,float)
        batch,time,input_dim=X.shape
        h=np.zeros((batch,self.Wh.shape[0])) if h0 is None else np.asarray(h0,float)
        states=[]
        outputs=[]
        for t in range(time):
            h=np.tanh(X[:,t]@self.Wx+h@self.Wh+self.bh)
            y=h@self.Wy+self.by
            states.append(h.copy()); outputs.append(y)
        return np.stack(outputs,axis=1),np.stack(states,axis=1)
    def predict_sequence(self,X):
        logits,_=self.forward(X)
        return np.argmax(logits,axis=-1)

def one_hot(indices,vocab_size):
    idx=np.asarray(indices,int)
    return np.eye(vocab_size)[idx]

def sequence_cross_entropy(targets,logits):
    targets=np.asarray(targets,int)
    probs=softmax(np.asarray(logits,float),axis=-1)
    flat_targets=targets.ravel()
    flat_probs=probs.reshape(-1,probs.shape[-1])
    return float(-np.mean(np.log(np.clip(flat_probs[np.arange(len(flat_targets)),flat_targets],1e-12,1))))

def truncated_bptt_windows(sequence,window):
    seq=np.asarray(sequence)
    return [seq[i:i+window] for i in range(0,len(seq)-1,window) if len(seq[i:i+window])>1]
