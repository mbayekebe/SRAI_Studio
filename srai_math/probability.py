import numpy as np
from scipy import stats
from srai_compat import dynamic_attribute

simulate_bernoulli = lambda probability=None, size=1, seed=42, p=None: np.random.default_rng(seed).binomial(1, probability if probability is not None else p, size)
def simulate_categorical(outcomes=None, probabilities=None, size=1, seed=42):
    if probabilities is None: probabilities,outcomes=outcomes,np.arange(len(outcomes))
    return np.random.default_rng(seed).choice(outcomes,size=size,p=probabilities)
bernoulli_pmf = lambda x,p: stats.bernoulli.pmf(x,p)
binomial_pmf = lambda x,n,p: stats.binom.pmf(x,n,p)
poisson_pmf = lambda x,mu: stats.poisson.pmf(x,mu)
normal_pdf = lambda x,mean=0,std=1: stats.norm.pdf(x,mean,std)
uniform_pdf = lambda x,a=0,b=1: stats.uniform.pdf(x,a,b-a)
exponential_pdf = lambda x,rate=1: stats.expon.pdf(x,scale=1/rate)
covariance = lambda x,y: float(np.cov(x,y,ddof=1)[0,1])
correlation = lambda x,y: float(np.corrcoef(x,y)[0,1])
covariance_matrix = lambda x: np.cov(x,rowvar=False)
running_mean = lambda x: np.cumsum(x)/np.arange(1,len(x)+1)

class FiniteProbabilitySpace:
    def __init__(self, probabilities): self.probabilities=probabilities
    def probability(self,event): return sum(self.probabilities.get(x,0) for x in event)
    def conditional_probability(self,a,b): return self.probability(set(a)&set(b))/self.probability(b)
    def independent(self,a,b): return np.isclose(self.probability(set(a)&set(b)),self.probability(a)*self.probability(b))
def empirical_probability(observations,event): return np.mean([x in event for x in observations])
def law_of_large_numbers_path(probability,size,seed=42): return running_mean(simulate_bernoulli(probability,size,seed))
def monte_carlo_expectation(function,sampler,samples): return float(np.mean(function(sampler(samples))))
def total_probability(priors,likelihoods): return float(np.dot(priors,likelihoods))
def bayes_discrete(priors,likelihoods):
    z=sum(priors[k]*likelihoods[k] for k in priors); return {k:priors[k]*likelihoods[k]/z for k in priors}
def bayes_binary(prior,sensitivity,specificity,positive=True):
    like1=sensitivity if positive else 1-sensitivity; like0=1-specificity if positive else specificity
    return prior*like1/(prior*like1+(1-prior)*like0)
probability_to_odds=lambda p:p/(1-p)
odds_to_probability=lambda o:o/(1+o)
posterior_odds=lambda prior_odds,likelihood_ratio:prior_odds*likelihood_ratio
def sequential_bayes(prior,evidence_likelihoods):
    out=[dict(prior)]; cur=dict(prior)
    for like in evidence_likelihoods: cur=bayes_discrete(cur,like); out.append(cur)
    return out
def sample_distribution(name,size,seed=42,**kw):
    r=np.random.default_rng(seed)
    if name=="normal": return r.normal(kw.get("mean",0),kw.get("std",1),size)
    if name=="poisson": return r.poisson(kw.get("rate",1),size)
    if name=="exponential": return r.exponential(1/kw.get("rate",1),size)
    if name=="uniform": return r.uniform(kw.get("low",0),kw.get("high",1),size)
def sample_moments(x): return {"mean":float(np.mean(x)),"variance":float(np.var(x)),"skewness":float(stats.skew(x))}
def empirical_cdf(samples,points): return np.searchsorted(np.sort(samples),points,side="right")/len(samples)
discrete_expectation=lambda values,probabilities:float(np.dot(values,probabilities))
def discrete_variance(values,probabilities):
    m=discrete_expectation(values,probabilities); return float(np.dot((np.asarray(values)-m)**2,probabilities))
def running_variance(x): return np.array([np.var(x[:i],ddof=1) if i>1 else 0 for i in range(1,len(x)+1)])
markov_bound=lambda expectation,threshold:min(1,expectation/threshold)
chebyshev_bound=lambda variance,deviation:min(1,variance/deviation**2)
hoeffding_bound=lambda epsilon,n,lower=0,upper=1:min(1,2*np.exp(-2*n*epsilon**2/(upper-lower)**2))
def sample_means(sampler,sample_size,repetitions): return np.asarray(sampler((repetitions,sample_size))).mean(axis=1)
sampling_standard_error=lambda population_std,sample_size:population_std/np.sqrt(sample_size)
standardize_sample_means=lambda means,population_mean,population_std,sample_size:(np.asarray(means)-population_mean)/(population_std/np.sqrt(sample_size))
finite_population_correction=lambda population_size,sample_size:np.sqrt((population_size-sample_size)/(population_size-1))
def bootstrap_statistic(data,statistic=np.mean,repetitions=1000,seed=42):
    r=np.random.default_rng(seed); x=np.asarray(data); return np.array([statistic(r.choice(x,len(x),replace=True)) for _ in range(repetitions)])
bootstrap_standard_error=lambda values:float(np.std(values,ddof=1))
__getattr__ = dynamic_attribute
