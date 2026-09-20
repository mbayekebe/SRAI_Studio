"""Deep Q-learning utilities."""
from __future__ import annotations
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self,capacity=10000,seed=42):
        self.capacity=capacity
        self.buffer=deque(maxlen=capacity)
        self.rng=np.random.default_rng(seed)
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((
            np.asarray(state,float),int(action),float(reward),
            np.asarray(next_state,float),bool(done)
        ))
    def sample(self,batch_size):
        if batch_size>len(self.buffer):
            raise ValueError("batch_size exceeds buffer size.")
        idx=self.rng.choice(len(self.buffer),batch_size,replace=False)
        batch=[self.buffer[i] for i in idx]
        s,a,r,s2,d=zip(*batch)
        return (np.asarray(s),np.asarray(a),np.asarray(r),
                np.asarray(s2),np.asarray(d,float))
    def __len__(self): return len(self.buffer)

class LinearDQN:
    def __init__(self,state_dim,n_actions,learning_rate=.05,gamma=.99,seed=42):
        rng=np.random.default_rng(seed)
        self.W=rng.normal(scale=.1,size=(state_dim,n_actions))
        self.target_W=self.W.copy()
        self.learning_rate=learning_rate
        self.gamma=gamma
    def q_values(self,states,target=False):
        W=self.target_W if target else self.W
        return np.asarray(states,float)@W
    def act(self,state,epsilon=.1,seed=42):
        rng=np.random.default_rng(seed)
        if rng.random()<epsilon:
            return int(rng.integers(self.W.shape[1]))
        return int(np.argmax(self.q_values(np.asarray(state)[None,:])[0]))
    def train_batch(self,states,actions,rewards,next_states,dones):
        q=self.q_values(states)
        target=rewards+self.gamma*(1-dones)*np.max(self.q_values(next_states,target=True),axis=1)
        pred=q[np.arange(len(actions)),actions]
        error=pred-target
        grad=np.zeros_like(q)
        grad[np.arange(len(actions)),actions]=error/len(actions)
        grad_W=np.asarray(states).T@grad
        self.W-=self.learning_rate*grad_W
        return float(np.mean(error**2))
    def update_target(self,tau=1.0):
        self.target_W=tau*self.W+(1-tau)*self.target_W

def epsilon_schedule(step,start=1.0,end=.05,decay=1000):
    return float(end+(start-end)*np.exp(-step/decay))
