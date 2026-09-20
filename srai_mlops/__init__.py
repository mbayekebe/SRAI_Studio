from dataclasses import dataclass
import numpy as np
from srai_compat import dynamic_attribute, set_seed
@dataclass
class InferencePayload: request_id:str; features:list
def build_inference_handler(predictor,version,feature_count):
    def handler(payload):
        if len(payload.features)!=feature_count: return {"request_id":payload.request_id,"error":"invalid feature count","version":version}
        return {"request_id":payload.request_id,"prediction":predictor(np.asarray(payload.features,float)),"version":version}
    return handler
openapi_stub=lambda title,version:{"openapi":"3.0.0","info":{"title":title,"version":version}}
def population_stability_index(reference,current,bins=10):
    edges=np.quantile(reference,np.linspace(0,1,bins+1)); edges[[0,-1]]=[-np.inf,np.inf]
    a=np.clip(np.histogram(reference,edges)[0]/len(reference),1e-6,None); b=np.clip(np.histogram(current,edges)[0]/len(current),1e-6,None)
    return float(np.sum((b-a)*np.log(b/a)))
drift_status=lambda psi:"high" if psi>=.25 else ("moderate" if psi>=.1 else "low")
performance_change=lambda baseline,current,higher_is_better=True:{"baseline":baseline,"current":current,"change":current-baseline,"degraded":current<baseline if higher_is_better else current>baseline}
def governance_report(model,version,owner,drift,performance,approved): return {"model":model,"version":version,"owner":owner,"drift":drift,"performance":performance,"approved":approved}
def subgroup_metric(frame,group,y_true,y_pred):
    return frame.groupby(group).apply(lambda x:float(np.mean(x[y_true]==x[y_pred])),include_groups=False).to_dict()
__getattr__ = dynamic_attribute
