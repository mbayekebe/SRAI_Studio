import math
import numpy as np
from scipy import stats
from srai_compat import dynamic_attribute

sample_mean = lambda x: float(np.mean(x))
sample_variance = lambda x: float(np.var(x, ddof=1))
standard_error_mean = lambda x: float(stats.sem(x))
one_sample_t_test = lambda x, null_mean=0, alternative="two-sided": {"statistic": stats.ttest_1samp(x, null_mean, alternative=alternative).statistic, "p_value": stats.ttest_1samp(x, null_mean, alternative=alternative).pvalue}
independent_t_test = lambda a, b, equal_variance=False, alternative="two-sided": {"statistic": stats.ttest_ind(a,b,equal_var=equal_variance,alternative=alternative).statistic, "p_value": stats.ttest_ind(a,b,equal_var=equal_variance,alternative=alternative).pvalue}
paired_t_test = lambda a, b, alternative="two-sided": {"statistic": stats.ttest_rel(a,b,alternative=alternative).statistic, "p_value": stats.ttest_rel(a,b,alternative=alternative).pvalue}
cohens_d_one_sample = lambda x, reference=0: float((np.mean(x)-reference)/np.std(x,ddof=1))
cohens_d_independent = lambda a,b,pooled=True: float((np.mean(a)-np.mean(b))/np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2)))
hedges_g = lambda a,b: cohens_d_independent(a,b)*(1-3/(4*(len(a)+len(b))-9))
risk_difference = lambda successes_treatment,total_treatment,successes_control,total_control: successes_treatment/total_treatment-successes_control/total_control
relative_risk = lambda successes_treatment,total_treatment,successes_control,total_control: (successes_treatment/total_treatment)/(successes_control/total_control)
bonferroni_alpha = lambda alpha, tests: alpha/tests
def normal_mean_confidence_interval(mean,population_std,sample_size,confidence=.95):
    z=stats.norm.ppf((1+confidence)/2); d=z*population_std/np.sqrt(sample_size); return mean-d,mean+d
def t_mean_confidence_interval(x,confidence=.95):
    x=np.asarray(x); d=stats.t.ppf((1+confidence)/2,len(x)-1)*stats.sem(x); return x.mean()-d,x.mean()+d
def proportion_confidence_interval_wilson(successes,trials,confidence=.95):
    z=stats.norm.ppf((1+confidence)/2); p=successes/trials; den=1+z*z/trials
    c=(p+z*z/(2*trials))/den; d=z*np.sqrt(p*(1-p)/trials+z*z/(4*trials**2))/den; return c-d,c+d
def bootstrap_confidence_interval(data,statistic=np.mean,confidence=.95,repetitions=1000,seed=42):
    r=np.random.default_rng(seed); x=np.asarray(data); v=[statistic(r.choice(x,len(x),replace=True)) for _ in range(repetitions)]
    return tuple(np.quantile(v,[(1-confidence)/2,(1+confidence)/2]))
required_sample_size_mean=lambda population_std,margin_error,confidence=.95:int(np.ceil((stats.norm.ppf((1+confidence)/2)*population_std/margin_error)**2))
required_sample_size_proportion=lambda margin_error,confidence=.95,anticipated_proportion=.5:int(np.ceil(stats.norm.ppf((1+confidence)/2)**2*anticipated_proportion*(1-anticipated_proportion)/margin_error**2))
bernoulli_mle=lambda x:float(np.mean(x))
bernoulli_log_likelihood=lambda p,x:float(np.sum(np.asarray(x)*np.log(p)+(1-np.asarray(x))*np.log(1-p)))
normal_mle=lambda x:(float(np.mean(x)),float(np.std(x)))
normal_log_likelihood=lambda mu,sigma,x:float(np.sum(stats.norm.logpdf(x,mu,sigma)))
poisson_mle=lambda x:float(np.mean(x))
poisson_log_likelihood=lambda rate,x:float(np.sum(stats.poisson.logpmf(x,rate)))
beta_bernoulli_posterior=lambda alpha,beta,successes,failures:(alpha+successes,beta+failures)
beta_posterior_mean=lambda alpha,beta:alpha/(alpha+beta)
map_beta_bernoulli=lambda alpha,beta,successes,failures:(alpha+successes-1)/(alpha+beta+successes+failures-2)
beta_credible_interval=lambda alpha,beta,confidence=.95:tuple(stats.beta.ppf([(1-confidence)/2,(1+confidence)/2],alpha,beta))
gamma_poisson_posterior=lambda shape,rate,total_count,exposure:(shape+total_count,rate+exposure)
gamma_posterior_mean=lambda shape,rate:shape/rate
__getattr__ = dynamic_attribute
