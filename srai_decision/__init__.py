import numpy as np, pandas as pd
from srai_compat import dynamic_attribute
def scenario_table(alternatives,scenarios,payoffs): return pd.DataFrame(payoffs,index=alternatives,columns=scenarios)
def regret_matrix(payoffs): return np.max(payoffs,axis=0)-np.asarray(payoffs)
def minimax_regret(payoffs,alternatives=None):
    i=int(np.argmin(regret_matrix(payoffs).max(axis=1))); return alternatives[i] if alternatives is not None else i
def robust_score(payoffs,probabilities=None,weight=.5):
    p=np.asarray(probabilities if probabilities is not None else np.ones(np.asarray(payoffs).shape[1])/np.asarray(payoffs).shape[1])
    a=np.asarray(payoffs); return weight*(a@p)+(1-weight)*a.min(axis=1)
def sequential_update(mean,variance,observation,observation_variance):
    gain=variance/(variance+observation_variance); return mean+gain*(observation-mean),(1-gain)*variance
threshold_policy=lambda value,threshold,below,above:above if value>=threshold else below
def epsilon_greedy(values,epsilon=.1,seed=42):
    r=np.random.default_rng(seed); return int(r.integers(len(values)) if r.random()<epsilon else np.argmax(values))
class SimpleDigitalTwin:
    def __init__(self,state,transition): self.state=dict(state); self.transition=transition
    def step(self,controls,disturbance): self.state=self.transition(self.state,controls,disturbance); return dict(self.state)
def twin_scenario(factory,periods,policies):
    out={}
    for name,policy in policies.items():
        twin=factory(); rows=[]
        for t in range(periods): rows.append(twin.step(policy(t,twin.state),{}))
        out[name]=__import__("pandas").DataFrame(rows)
    return out
calibration_error=lambda observed,predicted:float(np.mean(np.abs(np.asarray(observed)-np.asarray(predicted))))
__getattr__ = dynamic_attribute
