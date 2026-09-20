import numpy as np
import pandas as pd
def population_stability_index(reference,current,bins=10):
    ref=np.asarray(reference,float); cur=np.asarray(current,float)
    edges=np.quantile(ref,np.linspace(0,1,bins+1)); edges[0]=-np.inf; edges[-1]=np.inf
    rc,_=np.histogram(ref,bins=edges); cc,_=np.histogram(cur,bins=edges)
    rp=np.clip(rc/rc.sum(),1e-6,None); cp=np.clip(cc/cc.sum(),1e-6,None)
    return float(np.sum((cp-rp)*np.log(cp/rp)))
def drift_status(psi):
    return "significant" if psi>=.25 else "moderate" if psi>=.1 else "stable"
def performance_change(baseline,current,higher_is_better=True):
    delta=(current-baseline) if higher_is_better else (baseline-current)
    return {"delta":float(delta),"degraded":bool(delta<0)}
def subgroup_metric(frame,group_col,y_true_col,y_pred_col):
    rows=[]
    for group,part in frame.groupby(group_col):
        rows.append({"group":group,"accuracy":float(np.mean(part[y_true_col]==part[y_pred_col])),"n":len(part)})
    return pd.DataFrame(rows)
def governance_report(model_name,version,owner,drift,performance,approved):
    return {"model_name":model_name,"version":version,"owner":owner,"drift":drift,"performance":performance,"approved":bool(approved)}
