import numpy as np
def platform_scorecard(data_quality,forecast_accuracy,coverage,alert_timeliness,user_adoption):
    vals=[data_quality,forecast_accuracy,coverage,alert_timeliness,user_adoption]
    return {"data_quality":float(data_quality),"forecast_accuracy":float(forecast_accuracy),"coverage":float(coverage),"alert_timeliness":float(alert_timeliness),"user_adoption":float(user_adoption),"overall":float(np.mean(vals))}
def implementation_readiness(checks):
    failed=[k for k,v in checks.items() if not v]
    return {"ready":not failed,"failed_checks":failed,"completion_rate":float((len(checks)-len(failed))/max(len(checks),1))}
def intervention_priority(risk_score,population_exposed,cost):
    r=np.asarray(risk_score,float); p=np.asarray(population_exposed,float); c=np.asarray(cost,float)
    return r*p/np.where(c<=0,np.nan,c)
def capstone_architecture():
    return ["Data sources","Ingestion and validation","Lakehouse and metadata","Statistics and ML","Early-warning engine","RAG and policy assistant","Dashboards and alerts","Governance and audit"]
