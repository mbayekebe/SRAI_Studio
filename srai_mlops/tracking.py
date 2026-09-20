"""Lightweight experiment tracking."""
from __future__ import annotations
from pathlib import Path
import json
import time

class ExperimentRun:
    def __init__(self,name,output_dir):
        self.name=name
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.record={
            "name":name,
            "started_at":time.time(),
            "parameters":{},
            "metrics":{},
            "artifacts":[],
            "tags":{},
        }
    def log_params(self,**kwargs):
        self.record["parameters"].update(kwargs)
    def log_metrics(self,**kwargs):
        self.record["metrics"].update({k:float(v) for k,v in kwargs.items()})
    def log_artifact(self,path):
        self.record["artifacts"].append(str(path))
    def set_tags(self,**kwargs):
        self.record["tags"].update(kwargs)
    def finish(self):
        self.record["finished_at"]=time.time()
        path=self.output_dir/f"{self.name}.json"
        path.write_text(json.dumps(self.record,indent=2,sort_keys=True),encoding="utf-8")
        return path

def load_run(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def compare_runs(paths,metric,higher_is_better=True):
    rows=[]
    for path in paths:
        run=load_run(path)
        rows.append({
            "name":run["name"],
            "value":run["metrics"][metric],
            "path":str(path),
        })
    rows.sort(key=lambda x:x["value"],reverse=higher_is_better)
    return rows
