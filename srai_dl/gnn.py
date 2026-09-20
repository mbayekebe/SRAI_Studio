import numpy as np
def norm_adj(A):
    A=np.asarray(A,float)+np.eye(len(A)); d=A.sum(1); D=np.diag(1/np.sqrt(d)); return D@A@D
def gcn_layer(A,X,W): return norm_adj(A)@np.asarray(X,float)@np.asarray(W,float)
def graph_pool(X,mode="mean"):
    X=np.asarray(X,float)
    return X.mean(0) if mode=="mean" else X.max(0)
def message_passing(A,X):
    A=np.asarray(A,float); X=np.asarray(X,float)
    deg=np.clip(A.sum(1,keepdims=True),1,None)
    return A@X/deg
