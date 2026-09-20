"""Neural-network optimizers."""
from __future__ import annotations
import numpy as np

class SGD:
    def __init__(self,learning_rate=0.01,momentum=0.0):
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.velocity={}
    def step(self,params_grads):
        for i,(p,g) in enumerate(params_grads):
            v=self.velocity.get(i,np.zeros_like(p))
            v=self.momentum*v-self.learning_rate*g
            p+=v
            self.velocity[i]=v

class RMSProp:
    def __init__(self,learning_rate=0.001,decay=0.9,epsilon=1e-8):
        self.learning_rate=learning_rate; self.decay=decay; self.epsilon=epsilon
        self.avg={}
    def step(self,params_grads):
        for i,(p,g) in enumerate(params_grads):
            a=self.avg.get(i,np.zeros_like(p))
            a=self.decay*a+(1-self.decay)*g*g
            p-=self.learning_rate*g/(np.sqrt(a)+self.epsilon)
            self.avg[i]=a

class Adam:
    def __init__(self,learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.learning_rate=learning_rate; self.beta1=beta1; self.beta2=beta2; self.epsilon=epsilon
        self.m={}; self.v={}; self.t=0
    def step(self,params_grads):
        self.t+=1
        for i,(p,g) in enumerate(params_grads):
            m=self.m.get(i,np.zeros_like(p))
            v=self.v.get(i,np.zeros_like(p))
            m=self.beta1*m+(1-self.beta1)*g
            v=self.beta2*v+(1-self.beta2)*g*g
            mh=m/(1-self.beta1**self.t)
            vh=v/(1-self.beta2**self.t)
            p-=self.learning_rate*mh/(np.sqrt(vh)+self.epsilon)
            self.m[i]=m; self.v[i]=v

def clip_gradients(params_grads,max_norm):
    pairs=list(params_grads)
    total=np.sqrt(sum(np.sum(g*g) for _,g in pairs))
    if total>max_norm:
        scale=max_norm/(total+1e-12)
        pairs=[(p,g*scale) for p,g in pairs]
    return pairs,float(total)
