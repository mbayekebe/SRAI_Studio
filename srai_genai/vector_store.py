"""In-memory vector database."""
from __future__ import annotations
import numpy as np

class VectorStore:
    def __init__(self):
        self.vectors_=[]
        self.metadata_=[]
    def add(self,vectors,metadata):
        X=np.asarray(vectors,float)
        if len(X)!=len(metadata):
            raise ValueError("vectors and metadata length mismatch.")
        self.vectors_.extend(X)
        self.metadata_.extend(metadata)
        return self
    def _matrix(self):
        return np.asarray(self.vectors_,float)
    def search(self,query_vector,k=5,metric="cosine",filter_fn=None):
        X=self._matrix()
        q=np.asarray(query_vector,float)
        indices=np.arange(len(X))
        if filter_fn is not None:
            mask=np.array([bool(filter_fn(m)) for m in self.metadata_])
            X=X[mask]; indices=indices[mask]
        if len(X)==0:
            return []
        if metric=="cosine":
            Xn=X/np.clip(np.linalg.norm(X,axis=1,keepdims=True),1e-12,None)
            qn=q/np.clip(np.linalg.norm(q),1e-12,None)
            scores=Xn@qn
        elif metric=="euclidean":
            scores=-np.linalg.norm(X-q,axis=1)
        else:
            raise ValueError("Unsupported metric.")
        order=np.argsort(scores)[::-1][:k]
        return [{
            "index":int(indices[i]),
            "score":float(scores[i]),
            "metadata":self.metadata_[indices[i]],
        } for i in order]

def approximate_search(vectors,query,k=5,candidates=100,seed=42):
    X=np.asarray(vectors,float); q=np.asarray(query,float)
    rng=np.random.default_rng(seed)
    candidate_idx=rng.choice(len(X),min(candidates,len(X)),replace=False)
    subset=X[candidate_idx]
    scores=subset@q
    order=np.argsort(scores)[::-1][:k]
    return [(int(candidate_idx[i]),float(scores[i])) for i in order]

def vector_store_memory_bytes(n_vectors,dimension,dtype_bytes=4):
    return int(n_vectors*dimension*dtype_bytes)
