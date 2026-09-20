"""Policy-gradient and actor-critic utilities."""
from __future__ import annotations
import numpy as np

def softmax_policy(logits):
    x=np.asarray(logits,float)
    x=x-np.max(x,axis=-1,keepdims=True)
    e=np.exp(x)
    return e/e.sum(axis=-1,keepdims=True)

def sample_action(probabilities,seed=42):
    p=np.asarray(probabilities,float)
    return int(np.random.default_rng(seed).choice(len(p),p=p))

def discounted_returns(rewards,gamma=.99,normalize=False):
    out=[]; g=0.0
    for r in rewards[::-1]:
        g=float(r)+gamma*g
        out.append(g)
    result=np.asarray(out[::-1],float)
    if normalize and result.std()>1e-12:
        result=(result-result.mean())/result.std()
    return result

class LinearPolicy:
    def __init__(self,state_dim,n_actions,learning_rate=.05,seed=42):
        self.W=np.random.default_rng(seed).normal(scale=.1,size=(state_dim,n_actions))
        self.learning_rate=learning_rate
    def probabilities(self,states):
        return softmax_policy(np.asarray(states,float)@self.W)
    def update(self,states,actions,advantages):
        X=np.asarray(states,float)
        actions=np.asarray(actions,int)
        adv=np.asarray(advantages,float)
        P=self.probabilities(X)
        Y=np.eye(P.shape[1])[actions]
        grad=X.T@((P-Y)*adv[:,None])/len(X)
        self.W-=self.learning_rate*grad
        return float(-np.mean(np.log(np.clip(P[np.arange(len(actions)),actions],1e-12,1))*adv))

class LinearValue:
    def __init__(self,state_dim,learning_rate=.05):
        self.w=np.zeros(state_dim)
        self.learning_rate=learning_rate
    def predict(self,states):
        return np.asarray(states,float)@self.w
    def update(self,states,targets):
        X=np.asarray(states,float); y=np.asarray(targets,float)
        err=self.predict(X)-y
        self.w-=self.learning_rate*(X.T@err)/len(X)
        return float(np.mean(err**2))

def actor_critic_advantage(rewards,values,next_values,dones,gamma=.99):
    return np.asarray(rewards,float)+gamma*(1-np.asarray(dones,float))*np.asarray(next_values,float)-np.asarray(values,float)
