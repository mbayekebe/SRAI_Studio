"""Computational complexity and empirical benchmarking utilities."""
from __future__ import annotations
import time
import numpy as np

def operation_count_matrix_multiply(m,n,p):
    if min(m,n,p)<=0:
        raise ValueError("dimensions must be positive.")
    multiplications=m*n*p
    additions=m*p*(n-1)
    return {"multiplications":multiplications,"additions":additions,
            "total":multiplications+additions}

def empirical_runtime(function,sizes,repeats=3):
    results=[]
    for n in sizes:
        timings=[]
        for _ in range(repeats):
            start=time.perf_counter()
            function(int(n))
            timings.append(time.perf_counter()-start)
        results.append((int(n),float(np.median(timings))))
    return results

def estimate_loglog_slope(sizes,times):
    x=np.log(np.asarray(sizes,dtype=float))
    y=np.log(np.asarray(times,dtype=float))
    slope,intercept=np.polyfit(x,y,1)
    return float(slope),float(intercept)

def complexity_table():
    return [
        ("constant","O(1)"),
        ("logarithmic","O(log n)"),
        ("linear","O(n)"),
        ("linearithmic","O(n log n)"),
        ("quadratic","O(n^2)"),
        ("cubic","O(n^3)"),
        ("exponential","O(2^n)"),
    ]

def memory_bytes(shape,dtype=np.float64):
    return int(np.prod(shape)*np.dtype(dtype).itemsize)

def sparse_dense_memory(rows,cols,nonzeros,dtype_bytes=8,index_bytes=4):
    dense=rows*cols*dtype_bytes
    csr=(nonzeros*dtype_bytes)+(nonzeros*index_bytes)+((rows+1)*index_bytes)
    return {"dense_bytes":int(dense),"csr_bytes":int(csr),"compression_ratio":float(dense/csr)}
