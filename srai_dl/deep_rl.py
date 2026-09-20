import numpy as np
def epsilon_greedy(q,eps=.1,seed=42):
    rng=np.random.default_rng(seed)
    return int(rng.integers(len(q))) if rng.random()<eps else int(np.argmax(q))
def q_update(q,s,a,r,s2,alpha=.1,gamma=.99):
    q=np.asarray(q,float).copy(); q[s,a]+=alpha*(r+gamma*np.max(q[s2])-q[s,a]); return q
def discounted_returns(rewards,gamma=.99):
    out=[]; g=0
    for r in rewards[::-1]: g=r+gamma*g; out.append(g)
    return np.array(out[::-1],float)
class TinyDQN:
    def __init__(self,state_dim,actions,seed=42):
        rng=np.random.default_rng(seed); self.W=rng.normal(scale=.1,size=(state_dim,actions))
    def q_values(self,X): return np.asarray(X,float)@self.W
    def act(self,x): return int(np.argmax(self.q_values(np.asarray(x)[None,:])[0]))
