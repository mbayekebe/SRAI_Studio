import numpy as np
from srai_compat import dynamic_attribute

entropy = lambda p: float(-np.sum(np.where(np.asarray(p)>0,np.asarray(p)*np.log2(np.asarray(p)),0)))
binary_entropy = lambda p: entropy([p,1-p])
cross_entropy = lambda p,q: float(-np.sum(np.where(np.asarray(p)>0,np.asarray(p)*np.log2(np.asarray(q)),0)))
kl_divergence = lambda p,q: float(np.sum(np.where(np.asarray(p)>0,np.asarray(p)*np.log2(np.asarray(p)/np.asarray(q)),0)))
joint_entropy=lambda joint:entropy(np.asarray(joint).ravel())
def marginal_probabilities(joint): return np.asarray(joint).sum(axis=1),np.asarray(joint).sum(axis=0)
def mutual_information(joint):
    j=np.asarray(joint,float); r,c=marginal_probabilities(j); expected=r[:,None]*c[None,:]
    return float(np.sum(np.where(j>0,j*np.log2(j/expected),0)))
def conditional_entropy(joint,condition_on="columns"):
    r,c=marginal_probabilities(joint); return joint_entropy(joint)-entropy(c if condition_on=="columns" else r)
def information_gain(parent,children): return entropy(parent)-sum(w*entropy(p) for w,p in children)
__getattr__ = dynamic_attribute
