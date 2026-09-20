"""Transformer architecture comparison helpers."""
from __future__ import annotations
import numpy as np

def causal_mask(length):
    return np.tril(np.ones((length,length),dtype=bool))

def encoder_attention_mask(length):
    return np.ones((length,length),dtype=bool)

def parameter_estimate(vocab_size,d_model,layers,ff_dim,heads=8):
    embeddings=vocab_size*d_model
    attention_per_layer=4*d_model*d_model
    ff_per_layer=2*d_model*ff_dim
    norms_per_layer=4*d_model
    total=embeddings+layers*(attention_per_layer+ff_per_layer+norms_per_layer)
    return int(total)

def architecture_profile(name):
    profiles={
        "encoder":{"attention":"bidirectional","primary_tasks":["classification","retrieval","encoding"]},
        "decoder":{"attention":"causal","primary_tasks":["generation","completion","dialogue"]},
        "encoder_decoder":{"attention":"bidirectional + causal","primary_tasks":["translation","summarization","transduction"]},
    }
    return profiles[name]

def context_memory_bytes(context_length,d_model,layers,dtype_bytes=2):
    kv_cache=2*context_length*d_model*layers*dtype_bytes
    activations=context_length*d_model*dtype_bytes
    return {"kv_cache_bytes":int(kv_cache),"activation_bytes":int(activations)}
