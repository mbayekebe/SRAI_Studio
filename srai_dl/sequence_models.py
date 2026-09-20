"""Sequence-modeling helpers."""
from __future__ import annotations
import numpy as np

def create_next_step_dataset(series,window):
    x=np.asarray(series,float)
    X=[]; y=[]
    for i in range(len(x)-window):
        X.append(x[i:i+window])
        y.append(x[i+window])
    return np.asarray(X),np.asarray(y)

def teacher_forcing_inputs(sequence,start_token):
    seq=np.asarray(sequence)
    shifted=np.empty_like(seq)
    shifted[...,0]=start_token
    shifted[...,1:]=seq[...,:-1]
    return shifted

def mask_sequences(sequences,pad_value=0):
    seq=np.asarray(sequences)
    return (seq!=pad_value).astype(float)

def sequence_accuracy(targets,predictions,mask=None):
    y=np.asarray(targets)
    p=np.asarray(predictions)
    correct=(y==p).astype(float)
    if mask is None:
        return float(np.mean(correct))
    m=np.asarray(mask,float)
    return float(np.sum(correct*m)/np.sum(m))

def greedy_decode(step_function,start_token,end_token,max_length):
    tokens=[start_token]
    state=None
    for _ in range(max_length):
        token,state=step_function(tokens[-1],state)
        tokens.append(token)
        if token==end_token:
            break
    return tokens
