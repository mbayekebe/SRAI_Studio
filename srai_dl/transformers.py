from __future__ import annotations
import numpy as np
from .attention import multi_head_attention,scaled_dot_product_attention,causal_mask
def positional_encoding(length,d_model):
 p=np.arange(length)[:,None]; d=np.arange(d_model)[None,:]; a=p/(10000**(2*(d//2)/d_model)); e=np.empty((length,d_model)); e[:,0::2]=np.sin(a[:,0::2]); e[:,1::2]=np.cos(a[:,1::2]); return e
def layer_norm(X,epsilon=1e-5):
 X=np.asarray(X,float); return (X-X.mean(-1,keepdims=True))/np.sqrt(X.var(-1,keepdims=True)+epsilon)
def feed_forward(X,hidden_dim,seed=42):
 X=np.asarray(X,float); r=np.random.default_rng(seed); W1=r.normal(scale=np.sqrt(2/X.shape[-1]),size=(X.shape[-1],hidden_dim)); W2=r.normal(scale=np.sqrt(2/hidden_dim),size=(hidden_dim,X.shape[-1])); return np.maximum(X@W1,0)@W2
def transformer_encoder_block(X,num_heads=2,ff_dim=32,seed=42):
 X=np.asarray(X,float); a,_=multi_head_attention(X,X,X,num_heads,seed); x=layer_norm(X+a); return layer_norm(x+feed_forward(x,ff_dim,seed+1))
def transformer_decoder_self_attention(X,causal=True): return scaled_dot_product_attention(X,X,X,causal_mask(X.shape[-2]) if causal else None)
def parameter_count_transformer(d_model,ff_dim,num_heads,layers=1): return int(layers*(4*d_model*d_model+2*d_model*ff_dim+ff_dim+d_model+4*d_model))
