"""Convolution and pooling utilities."""
from __future__ import annotations
import numpy as np

def conv2d_single(image,kernel,stride=1,padding=0):
    image=np.asarray(image,float)
    kernel=np.asarray(kernel,float)
    if padding>0:
        image=np.pad(image,((padding,padding),(padding,padding)))
    kh,kw=kernel.shape
    oh=(image.shape[0]-kh)//stride+1
    ow=(image.shape[1]-kw)//stride+1
    out=np.empty((oh,ow),dtype=float)
    for i in range(oh):
        for j in range(ow):
            patch=image[i*stride:i*stride+kh,j*stride:j*stride+kw]
            out[i,j]=np.sum(patch*kernel)
    return out

def conv2d_batch(images,kernels,stride=1,padding=0):
    X=np.asarray(images,float)
    K=np.asarray(kernels,float)
    if X.ndim!=4 or K.ndim!=4:
        raise ValueError("images must be NCHW and kernels OIHW.")
    n,c,h,w=X.shape
    out_channels,in_channels,kh,kw=K.shape
    if c!=in_channels:
        raise ValueError("Channel mismatch.")
    if padding>0:
        X=np.pad(X,((0,0),(0,0),(padding,padding),(padding,padding)))
    oh=(X.shape[2]-kh)//stride+1
    ow=(X.shape[3]-kw)//stride+1
    out=np.zeros((n,out_channels,oh,ow))
    for b in range(n):
        for oc in range(out_channels):
            for i in range(oh):
                for j in range(ow):
                    patch=X[b,:,i*stride:i*stride+kh,j*stride:j*stride+kw]
                    out[b,oc,i,j]=np.sum(patch*K[oc])
    return out

def max_pool2d(images,pool_size=2,stride=None):
    X=np.asarray(images,float)
    if X.ndim!=4:
        raise ValueError("images must be NCHW.")
    stride=pool_size if stride is None else stride
    n,c,h,w=X.shape
    oh=(h-pool_size)//stride+1
    ow=(w-pool_size)//stride+1
    out=np.empty((n,c,oh,ow))
    for b in range(n):
        for ch in range(c):
            for i in range(oh):
                for j in range(ow):
                    patch=X[b,ch,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
                    out[b,ch,i,j]=np.max(patch)
    return out

def average_pool2d(images,pool_size=2,stride=None):
    X=np.asarray(images,float)
    stride=pool_size if stride is None else stride
    n,c,h,w=X.shape
    oh=(h-pool_size)//stride+1
    ow=(w-pool_size)//stride+1
    out=np.empty((n,c,oh,ow))
    for b in range(n):
        for ch in range(c):
            for i in range(oh):
                for j in range(ow):
                    patch=X[b,ch,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
                    out[b,ch,i,j]=np.mean(patch)
    return out

def flatten_features(X):
    X=np.asarray(X,float)
    return X.reshape(X.shape[0],-1)
