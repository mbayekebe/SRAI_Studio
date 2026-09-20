from __future__ import annotations
import numpy as np
def expected_decision_value(p,benefit,cost): p=np.asarray(p,float); return p*benefit-(1-p)*cost
def select_capacity_constrained(p,capacity,benefit=1.,cost=1.):
 value=expected_decision_value(p,benefit,cost); order=np.argsort(value)[::-1]; d=np.zeros(len(value),int); d[order[:capacity]]=1; return d,value
def subgroup_summary(y,p,g):
 y=np.asarray(y,int); p=np.asarray(p,int); g=np.asarray(g); out={}
 for z in np.unique(g):
  m=g==z; out[str(z)]={'n':int(m.sum()),'accuracy':float(np.mean(y[m]==p[m])),'selection_rate':float(np.mean(p[m]==1))}
 return out
def validate_pipeline_inputs(X,y):
 X=np.asarray(X); y=np.asarray(y); d={'matching_rows':len(X)==len(y),'finite_features':bool(np.all(np.isfinite(X))),'finite_target':bool(np.all(np.isfinite(y))),'nonempty':len(X)>0}; d['all_passed']=all(d.values()); return d
