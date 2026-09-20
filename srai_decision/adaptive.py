import numpy as np
def epsilon_greedy(values,epsilon=.1,seed=42):
    rng=np.random.default_rng(seed)
    return int(rng.integers(len(values))) if rng.random()<epsilon else int(np.argmax(values))
def update_running_mean(current_mean,count,new_value):
    return float(current_mean+(new_value-current_mean)/(count+1))
def threshold_policy(state,threshold,low_action,high_action):
    return high_action if state>=threshold else low_action
def sequential_update(prior_mean,prior_variance,observation,observation_variance):
    pp=1/prior_variance; po=1/observation_variance; pv=1/(pp+po)
    pm=pv*(pp*prior_mean+po*observation)
    return float(pm),float(pv)
def adaptive_policy_score(rewards):
    x=np.asarray(rewards,float)
    return {"cumulative_reward":float(x.sum()),"mean_reward":float(x.mean()),"last_period_reward":float(x[-1])}
