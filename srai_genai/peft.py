"""Parameter-efficient fine-tuning and LoRA."""
from __future__ import annotations
import numpy as np

class LoRALinear:
    def __init__(self,base_weight,rank=4,alpha=1.0,seed=42):
        self.W=np.asarray(base_weight,float).copy()
        self.rank=rank
        self.alpha=alpha
        rng=np.random.default_rng(seed)
        self.A=rng.normal(scale=.01,size=(self.W.shape[0],rank))
        self.B=np.zeros((rank,self.W.shape[1]))
    @property
    def scaling(self):
        return self.alpha/self.rank
    def effective_weight(self):
        return self.W+self.scaling*(self.A@self.B)
    def forward(self,X):
        return np.asarray(X,float)@self.effective_weight()
    def trainable_parameter_count(self):
        return int(self.A.size+self.B.size)
    def full_parameter_count(self):
        return int(self.W.size)

def lora_parameter_savings(input_dim,output_dim,rank):
    full=input_dim*output_dim
    lora=input_dim*rank+rank*output_dim
    return {
        "full_parameters":full,
        "lora_parameters":lora,
        "reduction_fraction":float(1-lora/full),
    }

def adapter_bottleneck(X,down_weight,up_weight):
    X=np.asarray(X,float)
    return np.maximum(X@np.asarray(down_weight,float),0)@np.asarray(up_weight,float)
