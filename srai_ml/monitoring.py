from __future__ import annotations
import numpy as np
def latency_summary(x):
 x=np.asarray(x,float); return {'mean_ms':float(x.mean()),'p50_ms':float(np.quantile(x,.5)),'p95_ms':float(np.quantile(x,.95)),'p99_ms':float(np.quantile(x,.99)),'max_ms':float(x.max())}
def error_rate(flags): return float(1-np.mean(np.asarray(flags,bool)))
def service_level_compliance(x,threshold_ms): return float(np.mean(np.asarray(x,float)<=threshold_ms))
def classification_monitoring_metrics(y,p):
 y=np.asarray(y,int); p=np.asarray(p,int); tp=np.sum((y==1)&(p==1)); pos=np.sum(y==1); pp=np.sum(p==1)
 return {'accuracy':float(np.mean(y==p)),'recall':float(tp/pos) if pos else 0.,'precision':float(tp/pp) if pp else 0.}
def fairness_gap(group_metrics,metric):
 v=[m[metric] for m in group_metrics.values()]; return float(max(v)-min(v)) if v else 0.
def monitoring_status(metrics,thresholds):
 out={}
 for k,v in metrics.items():
  if k in thresholds:
   r=thresholds[k]; out[k]='alert' if ((r['direction']=='max' and v>r['value']) or (r['direction']=='min' and v<r['value'])) else 'ok'
 return out
