"""Enterprise AI observability helpers."""
from __future__ import annotations
import numpy as np

def trace(request_id,service,latency_ms,status,tokens=0,cost=0):
    return {
        "request_id":request_id,
        "service":service,
        "latency_ms":float(latency_ms),
        "status":status,
        "tokens":int(tokens),
        "cost":float(cost),
    }

def aggregate_traces(traces):
    return {
        "requests":len(traces),
        "error_rate":sum(t["status"]!="ok" for t in traces)/max(len(traces),1),
        "mean_latency_ms":float(np.mean([t["latency_ms"] for t in traces])) if traces else 0,
        "total_tokens":sum(t["tokens"] for t in traces),
        "total_cost":sum(t["cost"] for t in traces),
    }

def service_level_objective(latencies_ms,target_ms,required_fraction=.95):
    x=np.asarray(latencies_ms,float)
    achieved=float(np.mean(x<=target_ms))
    return {"achieved_fraction":achieved,"met":achieved>=required_fraction}

def anomaly_flag(value,baseline_mean,baseline_std,z=3):
    return abs(value-baseline_mean)>z*baseline_std
