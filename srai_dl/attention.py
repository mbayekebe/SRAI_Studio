from __future__ import annotations
import numpy as np
from .activations import softmax
def scaled_dot_product_attention(Q,K,V,mask=None):
 Q=np.asarray(Q,float); K=np.asarray(K,float); V=np.asarray(V,float); s=Q@np.swapaxes(K,-1,-2)/np.sqrt(Q.shape[-1]); s=np.where(np.asarray(mask,bool),s,-1e9) if mask is not None else s; w=softmax(s,axis=-1); return w@V,w
def causal_mask(n): return np.tril(np.ones((n,n),bool))
def multi_head_attention(Q,K,V,num_heads,seed=42):
 Q=np.asarray(Q,float); K=np.asarray(K,float); V=np.asarray(V,float); d=Q.shape[-1]
 if d%num_heads: raise ValueError('dimension must divide heads')
 rng=np.random.default_rng(seed); h=d//num_heads; outs=[]; ws=[]
 for _ in range(num_heads):
  Wq=rng.normal(scale=.2,size=(d,h)); Wk=rng.normal(scale=.2,size=(d,h)); Wv=rng.normal(scale=.2,size=(d,h)); o,w=scaled_dot_product_attention(Q@Wq,K@Wk,V@Wv); outs.append(o); ws.append(w)
 return np.concatenate(outs,-1),np.stack(ws)
def attention_entropy(w):
 w=np.clip(np.asarray(w,float),1e-12,1); return -np.sum(w*np.log(w),axis=-1)
