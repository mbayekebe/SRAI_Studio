import numpy as np
def _allocate(raw,total):
    allocation=np.floor(raw).astype(int); remainder=total-allocation.sum()
    order=np.argsort(raw-allocation)[::-1]; allocation[order[:remainder]]+=1
    return allocation
def allocate_proportional(stratum_sizes,total_sample):
    s=np.asarray(stratum_sizes,float); return _allocate(s/s.sum()*total_sample,total_sample)
def allocate_neyman(stratum_sizes,stratum_sd,total_sample):
    N=np.asarray(stratum_sizes,float); S=np.asarray(stratum_sd,float)
    return _allocate(N*S/(N*S).sum()*total_sample,total_sample)
def design_weight(population_size,sample_size):
    N=np.asarray(population_size,float); n=np.asarray(sample_size,float)
    return np.divide(N,n,out=np.zeros_like(N),where=n>0)
def weighted_mean(values,weights):
    x=np.asarray(values,float); w=np.asarray(weights,float); return float(np.sum(x*w)/np.sum(w))
def effective_sample_size(weights):
    w=np.asarray(weights,float); return float((w.sum()**2)/np.sum(w**2))
def response_adjusted_weight(base_weight,response_rate):
    w=np.asarray(base_weight,float); r=np.asarray(response_rate,float)
    return np.divide(w,r,out=np.zeros_like(w),where=r>0)
