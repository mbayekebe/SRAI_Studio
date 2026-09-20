from __future__ import annotations
import numpy as np
from scipy import stats
def population_stability_index(reference,current,bins=10):
 r=np.asarray(reference,float); c=np.asarray(current,float); e=np.quantile(r,np.linspace(0,1,bins+1)); e[0]-=1e-9; e[-1]+=1e-9
 rp=np.clip(np.histogram(r,bins=e)[0]/len(r),1e-6,None); cp=np.clip(np.histogram(c,bins=e)[0]/len(c),1e-6,None)
 return float(np.sum((cp-rp)*np.log(cp/rp)))
def ks_drift_test(reference,current):
 z=stats.ks_2samp(reference,current); return float(z.statistic),float(z.pvalue)
def label_shift(reference,current):
 r=np.asarray(reference,int); c=np.asarray(current,int); out={}
 for k in np.union1d(np.unique(r),np.unique(c)):
  out[int(k)]={'reference_rate':float(np.mean(r==k)),'current_rate':float(np.mean(c==k)),'difference':float(np.mean(c==k)-np.mean(r==k))}
 return out
def rolling_metric(values,window):
 x=np.asarray(values,float); c=np.cumsum(np.insert(x,0,0.)); return (c[window:]-c[:-window])/window
def drift_alert(value,warning_threshold,critical_threshold): return 'critical' if value>=critical_threshold else ('warning' if value>=warning_threshold else 'normal')
