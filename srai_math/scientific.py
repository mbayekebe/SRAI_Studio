import numpy as np
from scipy import sparse
from srai_compat import dynamic_attribute
def standardize_columns(x):
    x=np.asarray(x,float); mean=x.mean(0); std=x.std(0); std=np.where(std==0,1,std); return (x-mean)/std,mean,std
pairwise_squared_distances=lambda x:np.maximum(np.sum(np.asarray(x)**2,1)[:,None]+np.sum(np.asarray(x)**2,1)[None,:]-2*np.asarray(x)@np.asarray(x).T,0)
to_csr=sparse.csr_matrix
sparse_matvec=lambda a,v:a@v
def chunked_mean(x,chunk_size=1000):
    x=np.asarray(x); return np.sum([part.sum(0) for part in np.array_split(x,np.ceil(len(x)/chunk_size))],axis=0)/len(x)
moving_average=lambda x,window:np.convolve(x,np.ones(window)/window,mode="valid")
__getattr__=dynamic_attribute
