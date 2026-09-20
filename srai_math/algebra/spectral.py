"""Eigenvalue, eigenvector, and spectral utilities."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .matrices import as_matrix
from .vectors import as_vector, l2_norm, normalize

def eigendecomposition(a: ArrayLike):
    A=as_matrix(a)
    if A.shape[0]!=A.shape[1]: raise ValueError("Eigendecomposition requires a square matrix.")
    return np.linalg.eig(A)

def symmetric_eigendecomposition(a: ArrayLike):
    A=as_matrix(a)
    if A.shape[0]!=A.shape[1]: raise ValueError("Eigendecomposition requires a square matrix.")
    if not np.allclose(A,A.T): raise ValueError("Matrix must be symmetric.")
    values,vectors=np.linalg.eigh(A)
    order=np.argsort(values)[::-1]
    return values[order],vectors[:,order]

def verify_eigenpair(a,eigenvalue,eigenvector,tol=1e-10):
    A=as_matrix(a); v=np.asarray(eigenvector,dtype=complex)
    if v.ndim!=1 or A.shape[1]!=v.shape[0]: raise ValueError("Matrix and vector dimensions do not agree.")
    return bool(np.linalg.norm(A@v-eigenvalue*v)<=tol)

def spectral_radius(a):
    values,_=eigendecomposition(a)
    return float(np.max(np.abs(values)))

def power_iteration(a,initial=None,max_iter=1000,tol=1e-10):
    A=as_matrix(a)
    if A.shape[0]!=A.shape[1]: raise ValueError("Power iteration requires a square matrix.")
    n=A.shape[0]
    v=np.ones(n) if initial is None else as_vector(initial)
    if v.shape[0]!=n: raise ValueError("Initial vector dimension does not agree with matrix.")
    v=normalize(v)
    old=0.0
    for iteration in range(1,max_iter+1):
        w=A@v
        if np.isclose(l2_norm(w),0.0): raise ValueError("Power iteration reached the zero vector.")
        v=normalize(w)
        value=float(v@A@v)
        if abs(value-old)<=tol:
            return value,v,iteration
        old=value
    return value,v,max_iter

def diagonalize(a,tol=1e-10):
    A=as_matrix(a)
    values,vectors=np.linalg.eig(A)
    if np.linalg.matrix_rank(vectors,tol=tol)<A.shape[0]:
        raise ValueError("Matrix is not diagonalizable.")
    D=np.diag(values); P=vectors; P_inv=np.linalg.inv(P)
    return P,D,P_inv

def matrix_power_via_eigendecomposition(a,power):
    if not isinstance(power,int) or power<0:
        raise ValueError("power must be a non-negative integer.")
    P,D,P_inv=diagonalize(a)
    return P@np.linalg.matrix_power(D,power)@P_inv
