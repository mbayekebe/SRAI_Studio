"""Benefits realization and AI value helpers."""
from __future__ import annotations
import numpy as np

def roi(benefits,costs):
    return float((benefits-costs)/costs) if costs else np.inf

def net_present_value(cash_flows,discount_rate):
    return float(sum(cf/((1+discount_rate)**t) for t,cf in enumerate(cash_flows)))

def benefit_register(benefits):
    return sorted(benefits,key=lambda x:x.get("value",0),reverse=True)

def value_realization_score(financial,service_quality,risk_reduction,capacity_building):
    values=[financial,service_quality,risk_reduction,capacity_building]
    return {
        "financial":float(financial),
        "service_quality":float(service_quality),
        "risk_reduction":float(risk_reduction),
        "capacity_building":float(capacity_building),
        "overall":float(np.mean(values)),
    }
