from __future__ import annotations
import json,time
from pathlib import Path
class ExperimentTracker:
 def __init__(self,run_name,output_dir):
  self.run_name=run_name; self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True)
  self.record_={'run_name':run_name,'start_time':time.time(),'parameters':{},'metrics':{},'artifacts':[]}
 def log_params(self,**kwargs): self.record_['parameters'].update(kwargs)
 def log_metrics(self,**kwargs): self.record_['metrics'].update({k:float(v) for k,v in kwargs.items()})
 def log_artifact(self,path): self.record_['artifacts'].append(str(path))
 def finish(self):
  self.record_['end_time']=time.time(); p=self.output_dir/f'{self.run_name}.json'; p.write_text(json.dumps(self.record_,indent=2,sort_keys=True)); return p
def reproducibility_manifest(seed,python_version,package_versions,data_signature):
 return {'seed':int(seed),'python_version':python_version,'package_versions':dict(package_versions),'data_signature':dict(data_signature)}
def compare_runs(a,b,metric):
 A=json.loads(Path(a).read_text()); B=json.loads(Path(b).read_text()); va=A['metrics'][metric]; vb=B['metrics'][metric]
 return {'run_a':A['run_name'],'run_b':B['run_name'],'metric':metric,'difference':float(vb-va),'better_run':B['run_name'] if vb>va else A['run_name']}
