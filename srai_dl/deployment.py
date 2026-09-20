"""Deep-learning deployment utilities."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def quantize_symmetric(weights,bits=8):
    w=np.asarray(weights,float)
    qmax=2**(bits-1)-1
    scale=np.max(np.abs(w))/qmax if np.max(np.abs(w))>0 else 1.0
    q=np.clip(np.round(w/scale),-qmax,qmax).astype(np.int8 if bits<=8 else np.int16)
    return q,float(scale)

def dequantize_symmetric(qweights,scale):
    return np.asarray(qweights,float)*scale

def prune_by_magnitude(weights,sparsity=.5):
    w=np.asarray(weights,float).copy()
    threshold=np.quantile(np.abs(w),sparsity)
    w[np.abs(w)<=threshold]=0.0
    return w

def model_size_bytes(parameters,dtype_bytes=4):
    return int(sum(np.asarray(p).size*dtype_bytes for p in parameters))

def export_model_metadata(name,version,input_shape,output_shape,framework,path):
    metadata={
        "name":name,"version":version,"input_shape":list(input_shape),
        "output_shape":list(output_shape),"framework":framework,
    }
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return path

def batch_inference(function,X,batch_size=32):
    X=np.asarray(X)
    outputs=[]
    for start in range(0,len(X),batch_size):
        outputs.append(np.asarray(function(X[start:start+batch_size])))
    return np.concatenate(outputs,axis=0)
