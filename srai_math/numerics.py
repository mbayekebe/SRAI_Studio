import numpy as np
from scipy.special import logsumexp
from srai_compat import dynamic_attribute

relative_error=lambda estimate,true: float(np.linalg.norm(np.asarray(estimate)-np.asarray(true))/(np.linalg.norm(true) or 1))
stable_logsumexp=logsumexp
stable_softmax=lambda x: np.exp(np.asarray(x)-logsumexp(x))
cholesky_factor=np.linalg.cholesky
householder_qr=lambda a: np.linalg.qr(np.asarray(a,float),mode="reduced")
modified_gram_schmidt=householder_qr
def stable_quadratic_roots(a,b,c):
    d=np.sqrt(b*b-4*a*c); q=-.5*(b+np.copysign(d,b))
    return np.array([q/a,c/q])
def conjugate_gradient(a,b,x0=None,tol=1e-10,max_iter=None):
    a=np.asarray(a,float); b=np.asarray(b,float)
    x=np.zeros_like(b) if x0 is None else np.asarray(x0,float).copy()
    r=b-a@x; p=r.copy(); rs=r@r; history=[np.sqrt(rs)]; max_iter=max_iter or len(b)*10
    for i in range(max_iter):
        ap=a@p; alpha=rs/(p@ap); x+=alpha*p; r-=alpha*ap
        new=r@r; history.append(np.sqrt(new))
        if np.sqrt(new)<tol: break
        p=r+(new/rs)*p; rs=new
    return x,history,i+1
def iterative_refinement(a,b,x0,iterations=5):
    a=np.asarray(a,float); b=np.asarray(b,float); x=np.asarray(x0,float).copy(); history=[]
    for _ in range(iterations):
        r=b-a@x; history.append(float(np.linalg.norm(r))); x+=np.linalg.solve(a,r)
    history.append(float(np.linalg.norm(b-a@x)))
    return x,history
__getattr__=dynamic_attribute
