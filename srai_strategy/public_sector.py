"""Public-sector AI implementation helpers."""
from __future__ import annotations

def service_prioritization(reach,impact,feasibility,risk):
    return float(.35*reach+.35*impact+.20*feasibility-.10*risk)

def administrative_readiness(legal_mandate,data_quality,process_maturity,owner,budget):
    checks={"legal_mandate":legal_mandate,"data_quality":data_quality,"process_maturity":process_maturity,"owner":owner,"budget":budget}
    return {"checks":checks,"ready":all(checks.values())}

def implementation_stage(stage):
    stages={
        "discover":["problem framing","stakeholder mapping","baseline"],
        "design":["service design","data design","risk assessment"],
        "pilot":["limited deployment","evaluation","user feedback"],
        "scale":["integration","capacity","operations"],
        "institutionalize":["policy","budget","governance","continuous improvement"],
    }
    return stages.get(stage,[])
