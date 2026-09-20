import numpy as np, pandas as pd
class SimpleDigitalTwin:
    def __init__(self,state,transition_function):
        self.state=dict(state); self.transition_function=transition_function; self.history=[dict(self.state)]
    def step(self,controls=None,disturbance=None):
        self.state=self.transition_function(dict(self.state),controls or {},disturbance or {})
        self.history.append(dict(self.state)); return dict(self.state)
    def run(self,steps,control_policy=None,disturbance_generator=None):
        for t in range(steps):
            c=control_policy(t,self.state) if control_policy else {}
            d=disturbance_generator(t,self.state) if disturbance_generator else {}
            self.step(c,d)
        return pd.DataFrame(self.history)
def calibration_error(observed,simulated):
    o=np.asarray(observed,float); s=np.asarray(simulated,float)
    return float(np.sqrt(np.mean((o-s)**2)))
def twin_scenario(twin_factory,steps,policies):
    return {name:twin_factory().run(steps,control_policy=policy) for name,policy in policies.items()}
