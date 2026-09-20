from dataclasses import dataclass
import time
@dataclass
class FeatureView:
    name:str
    entity_key:str
    features:list
    version:str="1"
class FeatureStore:
    def __init__(self):
        self.offline={}; self.online={}; self.views={}
    def register_view(self,view):
        self.views[view.name]=view; return self
    def write_offline(self,view_name,frame):
        self.offline[view_name]=frame.copy(); return self
    def materialize_online(self,view_name):
        frame=self.offline[view_name]; view=self.views[view_name]
        self.online[view_name]={row[view.entity_key]:{f:row[f] for f in view.features} for _,row in frame.iterrows()}
        return self
    def get_online_features(self,view_name,entity_id):
        return self.online.get(view_name,{}).get(entity_id)
class ModelRegistry:
    def __init__(self): self.models=[]
    def register(self,name,version,metrics,stage="development",metadata=None):
        record={"name":name,"version":version,"metrics":dict(metrics),"stage":stage,"metadata":metadata or {},"created_at":time.time()}
        self.models.append(record); return record
    def transition(self,name,version,new_stage,approved_by):
        for record in self.models:
            if record["name"]==name and record["version"]==version:
                record["stage"]=new_stage; record["approved_by"]=approved_by; return record
        raise KeyError("Model version not found")
    def latest(self,name,stage=None):
        rows=[r for r in self.models if r["name"]==name and (stage is None or r["stage"]==stage)]
        return rows[-1] if rows else None
