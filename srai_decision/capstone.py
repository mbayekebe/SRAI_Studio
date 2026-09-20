import numpy as np
def decision_intelligence_scorecard(framing,analytics,causal_evidence,optimization,governance):
    vals=[framing,analytics,causal_evidence,optimization,governance]
    return {"framing":float(framing),"analytics":float(analytics),"causal_evidence":float(causal_evidence),"optimization":float(optimization),"governance":float(governance),"overall":float(np.mean(vals))}
def decision_readiness(checks):
    failed=[k for k,v in checks.items() if not v]
    return {"ready":not failed,"failed_checks":failed,"completion_rate":float((len(checks)-len(failed))/max(len(checks),1))}
def decision_intelligence_architecture():
    return ["Decision framing","Data and evidence","Predictive analytics","Causal analysis","Simulation and scenarios","Optimization and prescriptions","Decision dashboard","Governance and learning"]
