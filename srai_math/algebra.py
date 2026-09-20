import numpy as np
from srai_compat import dynamic_attribute

dot = np.dot
cross = np.cross
transpose = np.transpose
trace = np.trace
rank = np.linalg.matrix_rank
determinant_2x2 = np.linalg.det
l1_norm = lambda x: np.linalg.norm(x, 1)
l2_norm = lambda x: np.linalg.norm(x, 2)
linf_norm = lambda x: np.linalg.norm(x, np.inf)
normalize = lambda x: np.asarray(x) / (np.linalg.norm(x) or 1)
distance = lambda a, b: np.linalg.norm(np.asarray(a) - np.asarray(b))
cosine_similarity = lambda a, b: float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1))
angle_between = lambda a,b: float(np.arccos(np.clip(cosine_similarity(a,b),-1,1)))
matrix_add = np.add
matrix_multiply = np.matmul
condition_number = np.linalg.cond
spectral_radius = lambda a: float(np.max(np.abs(np.linalg.eigvals(a))))
svd_decomposition = np.linalg.svd
eigendecomposition = np.linalg.eig
symmetric_eigendecomposition = np.linalg.eigh
reconstruct_from_svd = lambda u, s, vt: u @ np.diag(s) @ vt
projection_matrix = lambda a: np.asarray(a) @ np.linalg.pinv(np.asarray(a))
def project(x,basis):
    x=np.asarray(x); basis=np.asarray(basis)
    if basis.ndim==1: return (x@basis)/(basis@basis)*basis
    return projection_matrix(basis)@x
def residual(*args):
    if len(args)==3:
        a,x,b=args
        return np.asarray(b)-np.asarray(a)@np.asarray(x)
    x,estimate=args
    return np.asarray(x)-np.asarray(estimate)
reject = lambda x,basis:np.asarray(x)-project(x,basis)
verify_eigenpair = lambda a, value, vector: np.allclose(np.asarray(a) @ vector, value * np.asarray(vector))

class Vector:
    def __init__(self, values): self.values = tuple(map(float, values))
    @classmethod
    def from_iterable(cls, values): return cls(values)
    def __array__(self, dtype=None): return np.asarray(self.values, dtype=dtype)
    def __add__(self, other): return Vector(np.asarray(self.values) + np.asarray(other))
    def __mul__(self, scalar): return Vector(np.asarray(self.values) * scalar)
    __rmul__ = __mul__
    def __iter__(self): return iter(self.values)
    def __repr__(self): return f"Vector({list(self.values)})"

def gaussian_elimination(a, b): return np.linalg.solve(np.asarray(a, float), np.asarray(b, float))
def is_in_span(x, basis, tol=1e-9):
    b=np.asarray(basis,float)
    return np.linalg.norm(b@np.linalg.lstsq(b,np.asarray(x,float),rcond=None)[0]-x)<tol
def symmetric_eigendecomposition(a):
    values,vectors=np.linalg.eigh(a); order=np.argsort(values)[::-1]
    return values[order],vectors[:,order]
def rank_k_approximation(a,k):
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    return (u[:,:k]*s[:k])@vt[:k]
def frobenius_error(a,b): return float(np.linalg.norm(np.asarray(a)-np.asarray(b),"fro"))
def gram_schmidt(vectors):
    a=np.asarray(vectors,float); q,_=np.linalg.qr(a)
    return q
change_of_basis_matrix=lambda old_basis,new_basis:np.linalg.solve(np.asarray(new_basis),np.asarray(old_basis))
coordinates=lambda x,basis:np.linalg.lstsq(np.asarray(basis),np.asarray(x),rcond=None)[0]
reconstruct=lambda coordinates,basis:np.asarray(basis)@np.asarray(coordinates)
project_onto_subspace=lambda x,basis:projection_matrix(basis)@np.asarray(x)
column_space_basis=lambda a:np.linalg.svd(a,full_matrices=False)[0][:,:np.linalg.matrix_rank(a)]
row_space_basis=lambda a:np.linalg.svd(a,full_matrices=False)[2][:np.linalg.matrix_rank(a)]
null_space_basis=lambda a:__import__("scipy").linalg.null_space(a)
def diagonalize(a):
    values,p=np.linalg.eig(a)
    return p,np.diag(values),np.linalg.inv(p)
def matrix_power_via_eigendecomposition(a,n):
    p,d,p_inv=diagonalize(a)
    return p@np.linalg.matrix_power(d,n)@p_inv
def power_iteration(a,max_iter=1000,tol=1e-10):
    a=np.asarray(a,float); v=np.ones(a.shape[0]); v/=np.linalg.norm(v); old=0.0
    for i in range(max_iter):
        w=a@v; v=w/np.linalg.norm(w); value=float(v@a@v)
        if abs(value-old)<tol: break
        old=value
    return value,v,i+1
def explained_variance_ratio_from_singular_values(s):
    v=np.asarray(s)**2
    return v/v.sum()
def pca_fit_transform(x,n_components):
    x=np.asarray(x); mean=x.mean(0); _,s,vt=np.linalg.svd(x-mean,full_matrices=False)
    explained=(s[:n_components]**2)/(s**2).sum()
    return (x-mean)@vt[:n_components].T,vt[:n_components],explained,mean
def pca_inverse_transform(z,components,mean): return np.asarray(z)@np.asarray(components)+mean
__getattr__ = dynamic_attribute
