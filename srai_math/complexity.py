import time, numpy as np
from srai_compat import dynamic_attribute
complexity_table=lambda:[("Constant","O(1)"),("Logarithmic","O(log n)"),("Linear","O(n)"),("Quadratic","O(n²)")]
operation_count_matrix_multiply=lambda m,n,p:{"multiplications":m*n*p,"additions":m*p*(n-1)}
def empirical_runtime(function,sizes,repeats=3):
    out=[]
    for n in sizes:
        ts=[]
        for _ in range(repeats): t=time.perf_counter(); function(n); ts.append(time.perf_counter()-t)
        out.append((n,float(np.median(ts))))
    return out
def estimate_loglog_slope(x,y):
    result=np.polyfit(np.log(x),np.log(y),1); return float(result[0]),float(result[1])
memory_bytes=lambda shape,dtype=np.float64:int(np.prod(shape)*np.dtype(dtype).itemsize)
def sparse_dense_memory(rows,cols,nnz):
    dense=memory_bytes((rows,cols)); csr=nnz*12+(rows+1)*4
    return {"dense_bytes":dense,"csr_bytes":csr,"compression_ratio":dense/csr}
__getattr__=dynamic_attribute
