"""Enterprise AI governance platform helpers."""
from __future__ import annotations

def model_card(name,version,owner,purpose,limitations,metrics):
    return {
        "name":name,
        "version":version,
        "owner":owner,
        "purpose":purpose,
        "limitations":list(limitations),
        "metrics":dict(metrics),
    }

def risk_tier(impact,autonomy,data_sensitivity):
    score={"low":1,"medium":2,"high":3}[impact]+{"assistive":1,"semi":2,"autonomous":3}[autonomy]+{"low":1,"medium":2,"high":3}[data_sensitivity]
    return "critical" if score>=8 else "high" if score>=6 else "medium" if score>=4 else "low"

def approval_workflow(required_approvals,received_approvals):
    missing=[a for a in required_approvals if a not in received_approvals]
    return {"approved":not missing,"missing":missing}

def governance_register(records):
    return sorted(records,key=lambda x:(x["risk_tier"],x["name"]))
