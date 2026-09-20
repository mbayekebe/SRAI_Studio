"""AI platform engineering helpers."""
from __future__ import annotations
import numpy as np

def platform_capabilities():
    return [
        "self_service_environments",
        "model_registry",
        "feature_store",
        "vector_store",
        "workflow_orchestration",
        "observability",
        "security",
        "governance",
    ]

def tenancy_plan(teams,isolation="namespace"):
    return [{"team":team,"isolation":isolation,"quota":"standard"} for team in teams]

def capacity_plan(requests_per_second,average_latency_seconds,headroom=.3):
    base=requests_per_second*average_latency_seconds
    return {
        "base_concurrency":base,
        "recommended_concurrency":int(np.ceil(base*(1+headroom))),
    }

def golden_path_readiness(template,ci,security,monitoring,documentation):
    checks={"template":template,"ci":ci,"security":security,"monitoring":monitoring,"documentation":documentation}
    return {"checks":checks,"ready":all(checks.values())}
