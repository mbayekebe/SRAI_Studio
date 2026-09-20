import numpy as np
from srai_compat import dynamic_attribute
def normalized_scores(x):
    x=np.asarray(x,float); std=x.std(0)
    return (x-x.mean(0))/np.where(std==0,1,std)
weighted_score=lambda scores,weights:np.asarray(scores)@np.asarray(weights)
def rank_options(scores,labels): return sorted(zip(labels,np.asarray(scores)),key=lambda x:x[1],reverse=True)
scenario_loss=lambda probabilities,losses:float(np.dot(probabilities,losses))
portfolio_risk=lambda weights,covariance:float(np.sqrt(np.asarray(weights)@np.asarray(covariance)@np.asarray(weights)))
__getattr__=dynamic_attribute
