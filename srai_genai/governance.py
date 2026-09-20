from __future__ import annotations

def risk_classification(impact,autonomy,data_sensitivity):
    score={"low":1,"medium":2,"high":3}[impact]+{"assistive":1,"semi_autonomous":2,"autonomous":3}[autonomy]+{"low":1,"medium":2,"high":3}[data_sensitivity]
    return "critical" if score>=8 else "high" if score>=6 else "medium" if score>=4 else "low"

def governance_controls(risk_level):
    base=["model_card","data_lineage","evaluation_record","owner"]
    add={"low":[],"medium":["periodic_review","human_override"],"high":["independent_validation","continuous_monitoring","incident_plan"],"critical":["executive_approval","mandatory_human_decision","external_audit"]}
    levels=["low","medium","high","critical"]; controls=list(base)
    for level in levels[:levels.index(risk_level)+1]: controls.extend(add[level])
    return list(dict.fromkeys(controls))

def approval_gate(metrics,thresholds):
    failures={}
    for name,rule in thresholds.items():
        value=metrics.get(name)
        if value is None: failures[name]="missing"
        elif rule["direction"]=="min" and value<rule["value"]: failures[name]=value
        elif rule["direction"]=="max" and value>rule["value"]: failures[name]=value
    return {"approved":not failures,"failures":failures}

def audit_record(system,version,owner,risk_level,decision):
    return {"system":system,"version":version,"owner":owner,"risk_level":risk_level,"decision":decision}
