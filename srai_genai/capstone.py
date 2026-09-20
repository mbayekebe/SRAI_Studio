from __future__ import annotations
import numpy as np

def enterprise_ai_scorecard(retrieval_quality,groundedness,safety,latency,cost):
    vals=[retrieval_quality,groundedness,safety,latency,cost]
    return {"retrieval_quality":float(retrieval_quality),"groundedness":float(groundedness),"safety":float(safety),"latency":float(latency),"cost":float(cost),"overall":float(np.mean(vals))}

def choose_model_route(sensitivity,complexity,budget_priority):
    if sensitivity=="high": return "private-hosted-model"
    if budget_priority=="high" and complexity=="low": return "small-efficient-model"
    if complexity=="high": return "advanced-reasoning-model"
    return "general-purpose-model"

def capstone_readiness(checks):
    failed=[k for k,v in checks.items() if not v]
    return {"ready":not failed,"failed_checks":failed,"completion_rate":float((len(checks)-len(failed))/max(len(checks),1))}
