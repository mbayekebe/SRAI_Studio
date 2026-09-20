"""Numerical linear algebra utilities."""
from __future__ import annotations
import numpy as np

def modified_gram_schmidt(A,tol=1e-12):
    A=np.asarray(A,dtype=float)
    m,n=A.shape
    Q=np.zeros((m,n)); R=np.zeros((n,n))
    for j in range(n):
        v=A[:,j].copy()
        for i in range(j):
            R[i,j]=Q[:,i]@v
            v-=R[i,j]*Q[:,i]
        R[j,j]=np.linalg.norm(v)
        if R[j,j]<=tol:
            raise ValueError("Columns are linearly dependent.")
        Q[:,j]=v/R[j,j]
    return Q,R

def householder_qr(A):
    A=np.asarray(A,dtype=float)
    m,n=A.shape
    Q=np.eye(m); R=A.copy()
    for k in range(min(m,n)):
        x=R[k:,k]
        norm=np.linalg.norm(x)
        if np.isclose(norm,0): continue
        sign=-1.0 if x[0]<0 else 1.0
        v=x.copy(); v[0]+=sign*norm
        v/=np.linalg.norm(v)
        H=np.eye(m)
        H[k:,k:]-=2*np.outer(v,v)
        R=H@R
        Q=Q@H
    # Return the reduced factorization so its interface matches modified
    # Gram–Schmidt for rectangular m-by-n inputs.
    return Q[:, :n], R[:n, :]

def cholesky_factor(A):
    A=np.asarray(A,dtype=float)
    if not np.allclose(A,A.T):
        raise ValueError("Matrix must be symmetric.")
    return np.linalg.cholesky(A)

def conjugate_gradient(A,b,x0=None,tol=1e-10,max_iter=None):
    A=np.asarray(A,dtype=float); b=np.asarray(b,dtype=float)
    n=b.size
    x=np.zeros(n) if x0 is None else np.asarray(x0,dtype=float).copy()
    max_iter=n*10 if max_iter is None else max_iter
    r=b-A@x; p=r.copy(); rs=r@r
    history=[np.sqrt(rs)]
    for k in range(max_iter):
        Ap=A@p
        alpha=rs/(p@Ap)
        x+=alpha*p
        r-=alpha*Ap
        new=r@r
        history.append(np.sqrt(new))
        if np.sqrt(new)<tol:
            return x,history,k+1
        p=r+(new/rs)*p
        rs=new
    return x,history,max_iter

def iterative_refinement(A,b,x=None,iterations=3):
    A=np.asarray(A,dtype=float); b=np.asarray(b,dtype=float)
    x=np.linalg.solve(A,b) if x is None else np.asarray(x,dtype=float).copy()
    history=[float(np.linalg.norm(b-A@x))]
    for _ in range(iterations):
        r=b-A@x
        correction=np.linalg.solve(A,r)
        x=x+correction
        history.append(float(np.linalg.norm(b-A@x)))
    return x,history
