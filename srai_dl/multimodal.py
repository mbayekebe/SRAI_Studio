import numpy as np
def concatenate_fusion(*modalities): return np.concatenate([np.asarray(m,float) for m in modalities],axis=1)
def gated_fusion(A,B,gate):
    g=1/(1+np.exp(-np.asarray(gate,float))); return g*np.asarray(A,float)+(1-g)*np.asarray(B,float)
def cosine_alignment(A,B):
    A=np.asarray(A,float); B=np.asarray(B,float)
    return float(np.mean(np.sum(A*B,1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-12)))
