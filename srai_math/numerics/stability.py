"""Floating-point and stability utilities."""
from __future__ import annotations
import math
import numpy as np

def stable_logsumexp(x):
    values=np.asarray(x,dtype=float)
    m=np.max(values)
    return float(m+np.log(np.sum(np.exp(values-m))))

def stable_softmax(x):
    values=np.asarray(x,dtype=float)
    shifted=values-np.max(values)
    e=np.exp(shifted)
    return e/e.sum()

def stable_quadratic_roots(a,b,c):
    if a==0:
        raise ValueError("a must be nonzero.")
    disc=b*b-4*a*c
    if disc<0:
        return np.roots([a,b,c])
    s=math.sqrt(disc)
    q=-0.5*(b+math.copysign(s,b))
    x1=q/a
    x2=c/q if q!=0 else -b/a-x1
    return np.array([x1,x2],dtype=float)

def kahan_sum(x):
    total=0.0; compensation=0.0
    for value in np.asarray(x,dtype=float):
        y=value-compensation
        temp=total+y
        compensation=(temp-total)-y
        total=temp
    return float(total)

def relative_error(approximation,truth,epsilon=1e-15):
    return float(abs(approximation-truth)/max(abs(truth),epsilon))

def machine_summary():
    info=np.finfo(float)
    return {"epsilon":float(info.eps),"tiny":float(info.tiny),"max":float(info.max)}
