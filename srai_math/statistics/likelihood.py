"""Maximum likelihood and elementary Bayesian inference utilities."""
from __future__ import annotations
import math
import numpy as np
from scipy import stats

def _vector(data):
    x=np.asarray(data,dtype=float)
    if x.ndim!=1 or x.size==0: raise ValueError("data must be a non-empty vector.")
    return x

def bernoulli_log_likelihood(p,data):
    if not 0<p<1: return float("-inf")
    x=_vector(data)
    if not np.all(np.isin(x,[0.,1.])): raise ValueError("data must be binary.")
    return float(np.sum(x*np.log(p)+(1-x)*np.log(1-p)))

def bernoulli_mle(data):
    x=_vector(data)
    if not np.all(np.isin(x,[0.,1.])): raise ValueError("data must be binary.")
    return float(x.mean())

def normal_log_likelihood(mean,std,data):
    if std<=0: return float("-inf")
    return float(np.sum(stats.norm.logpdf(_vector(data),loc=mean,scale=std)))

def normal_mle(data):
    x=_vector(data)
    return float(x.mean()),float(x.std(ddof=0))

def poisson_log_likelihood(rate,data):
    if rate<=0: return float("-inf")
    x=_vector(data)
    if np.any(x<0) or np.any(x!=np.floor(x)): raise ValueError("data must be counts.")
    return float(np.sum(x*np.log(rate)-rate-np.vectorize(math.lgamma)(x+1)))

def poisson_mle(data):
    x=_vector(data)
    if np.any(x<0) or np.any(x!=np.floor(x)): raise ValueError("data must be counts.")
    return float(x.mean())

def beta_bernoulli_posterior(alpha,beta,successes,failures):
    if alpha<=0 or beta<=0 or successes<0 or failures<0: raise ValueError("Invalid inputs.")
    return float(alpha+successes),float(beta+failures)

def beta_posterior_mean(alpha,beta):
    if alpha<=0 or beta<=0: raise ValueError("Parameters must be positive.")
    return float(alpha/(alpha+beta))

def beta_credible_interval(alpha,beta,credibility=0.95):
    if alpha<=0 or beta<=0 or not 0<credibility<1: raise ValueError("Invalid inputs.")
    q=(1-credibility)/2
    lo,hi=stats.beta.ppf([q,1-q],alpha,beta)
    return float(lo),float(hi)

def map_beta_bernoulli(alpha,beta,successes,failures):
    a,b=beta_bernoulli_posterior(alpha,beta,successes,failures)
    if a<=1 or b<=1: raise ValueError("Interior MAP requires posterior parameters > 1.")
    return float((a-1)/(a+b-2))

def gamma_poisson_posterior(shape,rate,total_count,exposure):
    if shape<=0 or rate<=0 or total_count<0 or exposure<=0: raise ValueError("Invalid inputs.")
    return float(shape+total_count),float(rate+exposure)

def gamma_posterior_mean(shape,rate):
    if shape<=0 or rate<=0: raise ValueError("Parameters must be positive.")
    return float(shape/rate)
