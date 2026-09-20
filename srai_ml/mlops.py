from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

def hash_array(a): return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def dataset_signature(X,y=None):
 X=np.asarray(X); d={'shape':tuple(X.shape),'dtype':str(X.dtype),'hash':hash_array(X)}
 if y is not None:
  y=np.asarray(y); d.update(target_shape=tuple(y.shape),target_hash=hash_array(y))
 return d
def model_card(name,version,metrics,features,limitations,intended_use):
 return {'name':name,'version':version,'metrics':dict(metrics),'features':list(features),'limitations':list(limitations),'intended_use':intended_use}
def save_json_artifact(data,path):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data,indent=2,sort_keys=True)); return p
def deployment_checklist():
 return ['data_schema_validated','training_data_versioned','model_artifact_versioned','metrics_approved','fairness_reviewed','rollback_plan_defined','monitoring_enabled','owners_assigned']
