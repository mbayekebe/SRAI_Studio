"""AI transformation-roadmap helpers."""
from __future__ import annotations
import pandas as pd

def transformation_roadmap(initiatives):
    frame=pd.DataFrame(initiatives)
    order={"Mobilize":1,"Foundation":2,"Pilot":3,"Scale":4,"Institutionalize":5}
    frame["phase_order"]=frame["phase"].map(order)
    return frame.sort_values(["phase_order","priority"],ascending=[True,False]).drop(columns="phase_order")

def dependency_check(initiatives):
    names={i["name"] for i in initiatives}
    missing=[]
    for initiative in initiatives:
        for dependency in initiative.get("dependencies",[]):
            if dependency not in names:
                missing.append({"initiative":initiative["name"],"missing_dependency":dependency})
    return {"valid":not missing,"missing":missing}

def milestone_status(completed,total):
    ratio=completed/max(total,1)
    return "complete" if ratio==1 else "on_track" if ratio>=.7 else "at_risk" if ratio>=.4 else "critical"
