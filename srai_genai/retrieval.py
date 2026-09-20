"""Embedding and retrieval foundations."""
from __future__ import annotations
import re
import numpy as np

def bag_of_words_embeddings(texts,vocabulary=None):
    tokenized=[re.findall(r"\b\w+\b",str(t).lower()) for t in texts]
    vocab=sorted(set(tok for doc in tokenized for tok in doc)) if vocabulary is None else list(vocabulary)
    index={t:i for i,t in enumerate(vocab)}
    X=np.zeros((len(texts),len(vocab)))
    for i,doc in enumerate(tokenized):
        for token in doc:
            if token in index: X[i,index[token]]+=1
    return X,vocab

def normalize_embeddings(X):
    X=np.asarray(X,float)
    norms=np.linalg.norm(X,axis=1,keepdims=True)
    return X/np.clip(norms,1e-12,None)

def cosine_search(query_embedding,document_embeddings,k=3):
    q=np.asarray(query_embedding,float)
    D=normalize_embeddings(document_embeddings)
    q=q/np.clip(np.linalg.norm(q),1e-12,None)
    scores=D@q
    order=np.argsort(scores)[::-1][:k]
    return [(int(i),float(scores[i])) for i in order]

def chunk_text(text,chunk_size=50,overlap=10):
    tokens=str(text).split()
    if chunk_size<=0 or overlap>=chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks=[]
    step=chunk_size-overlap
    for start in range(0,len(tokens),step):
        chunk=tokens[start:start+chunk_size]
        if chunk: chunks.append(" ".join(chunk))
        if start+chunk_size>=len(tokens): break
    return chunks

def reciprocal_rank_fusion(rankings,k=60):
    scores={}
    for ranking in rankings:
        for rank,item in enumerate(ranking,1):
            scores[item]=scores.get(item,0)+1/(k+rank)
    return sorted(scores.items(),key=lambda x:x[1],reverse=True)
