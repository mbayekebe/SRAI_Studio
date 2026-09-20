"""Core optimization utilities."""
from __future__ import annotations
import numpy as np

def is_convex_quadratic(matrix, tol=1e-10):
    A=np.asarray(matrix,dtype=float)
    if A.ndim!=2 or A.shape[0]!=A.shape[1]:
        raise ValueError("matrix must be square.")
    if not np.allclose(A,A.T):
        return False
    return bool(np.all(np.linalg.eigvalsh(A)>=-tol))

def quadratic_value(x,Q,c=None,bias=0.0):
    x=np.asarray(x,dtype=float)
    Q=np.asarray(Q,dtype=float)
    c=np.zeros_like(x) if c is None else np.asarray(c,dtype=float)
    return float(0.5*x@Q@x+c@x+bias)

def quadratic_gradient(x,Q,c=None):
    x=np.asarray(x,dtype=float)
    Q=np.asarray(Q,dtype=float)
    c=np.zeros_like(x) if c is None else np.asarray(c,dtype=float)
    return Q@x+c

def project_box(x,lower,upper):
    return np.clip(np.asarray(x,dtype=float),lower,upper)

def project_simplex(v,total=1.0):
    x=np.asarray(v,dtype=float)
    if x.ndim!=1 or total<=0:
        raise ValueError("v must be a vector and total positive.")
    u=np.sort(x)[::-1]
    cssv=np.cumsum(u)-total
    rho=np.nonzero(u-cssv/np.arange(1,len(u)+1,dtype=float)>0)[0]
    if rho.size==0:
        return np.full_like(x,total/len(x))
    r=rho[-1]
    theta=cssv[r]/(r+1)
    return np.maximum(x-theta,0.0)

def constraint_violation(A,b,x):
    A=np.asarray(A,dtype=float); b=np.asarray(b,dtype=float); x=np.asarray(x,dtype=float)
    return np.maximum(A@x-b,0.0)

def lagrangian(objective_value,constraint_values,multipliers):
    g=np.asarray(constraint_values,dtype=float)
    lam=np.asarray(multipliers,dtype=float)
    return float(objective_value+lam@g)

def kkt_residual(gradient_value,A,x,b,multipliers):
    grad=np.asarray(gradient_value,dtype=float)
    A=np.asarray(A,dtype=float); x=np.asarray(x,dtype=float)
    b=np.asarray(b,dtype=float); lam=np.asarray(multipliers,dtype=float)
    stationarity=np.linalg.norm(grad+A.T@lam)
    primal=np.linalg.norm(np.maximum(A@x-b,0.0))
    dual=np.linalg.norm(np.minimum(lam,0.0))
    comp=np.linalg.norm(lam*(A@x-b))
    return {"stationarity":float(stationarity),"primal":float(primal),
            "dual":float(dual),"complementarity":float(comp)}
