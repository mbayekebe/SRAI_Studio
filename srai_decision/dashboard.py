import pandas as pd
def decision_kpis(expected_value,risk,implementation_cost,coverage):
    return {"expected_value":float(expected_value),"risk":float(risk),"implementation_cost":float(implementation_cost),"coverage":float(coverage)}
def traffic_light(value,green_threshold,amber_threshold,higher_is_better=True):
    if higher_is_better:
        return "green" if value>=green_threshold else "amber" if value>=amber_threshold else "red"
    return "green" if value<=green_threshold else "amber" if value<=amber_threshold else "red"
def decision_register(rows):
    f=pd.DataFrame(rows)
    return f.sort_values("priority",ascending=False) if "priority" in f.columns else f
def dashboard_readiness(data_fresh,model_validated,decision_owner_assigned,risks_documented):
    checks={"data_fresh":bool(data_fresh),"model_validated":bool(model_validated),"decision_owner_assigned":bool(decision_owner_assigned),"risks_documented":bool(risks_documented)}
    return {"checks":checks,"ready":all(checks.values())}
