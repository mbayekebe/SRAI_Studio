import numpy as np
from .matrices import as_matrix
from .vectors import as_vector

def _threshold(A,s):
    return max(A.shape)*np.finfo(float).eps*(s[0] if s.size else 0.0)

def column_space_basis(a,tol=None):
    A=as_matrix(a); U,s,_=np.linalg.svd(A,full_matrices=False)
    if tol is None: tol=_threshold(A,s)
    return U[:,:int(np.sum(s>tol))]

def row_space_basis(a,tol=None):
    A=as_matrix(a); _,s,Vt=np.linalg.svd(A,full_matrices=False)
    if tol is None: tol=_threshold(A,s)
    return Vt[:int(np.sum(s>tol)),:]

def null_space_basis(a,tol=None):
    A=as_matrix(a); _,s,Vt=np.linalg.svd(A,full_matrices=True)
    if tol is None: tol=_threshold(A,s)
    return Vt[int(np.sum(s>tol)):,:].T

def coordinates(v,basis):
    x=as_vector(v); B=as_matrix(basis)
    if B.shape[0]!=x.shape[0]: raise ValueError("Basis and vector dimensions do not agree.")
    c,_,r,_=np.linalg.lstsq(B,x,rcond=None)
    if r<B.shape[1] or not np.allclose(B@c,x):
        raise ValueError("Vector is not uniquely representable in the supplied basis.")
    return c

def reconstruct(coefficients,basis):
    c=as_vector(coefficients); B=as_matrix(basis)
    if B.shape[1]!=c.shape[0]: raise ValueError("Coefficient count must match basis size.")
    return B@c

def projection_matrix(basis):
    B=as_matrix(basis)
    if B.shape[1]==0: return np.zeros((B.shape[0],B.shape[0]))
    Q,_=np.linalg.qr(B,mode="reduced")
    Q=Q[:,:np.linalg.matrix_rank(B)]
    return Q@Q.T

def project_onto_subspace(v,basis):
    x=as_vector(v); P=projection_matrix(basis)
    if P.shape[0]!=x.shape[0]: raise ValueError("Basis and vector dimensions do not agree.")
    return P@x

def is_in_span(v,basis,tol=1e-10):
    x=as_vector(v)
    return bool(np.linalg.norm(x-project_onto_subspace(x,basis))<=tol)

def change_of_basis_matrix(old_basis,new_basis):
    B_old=as_matrix(old_basis); B_new=as_matrix(new_basis)
    if B_old.shape!=B_new.shape or B_old.shape[0]!=B_old.shape[1]:
        raise ValueError("Both bases must be square and equally shaped.")
    return np.linalg.solve(B_new,B_old)
