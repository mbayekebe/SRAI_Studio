from dataclasses import dataclass
import numpy as np
from srai_compat import dynamic_attribute
@dataclass
class Layer: name:str; purpose:str; components:list
def reference_layers(): return [Layer("Data","Trusted agricultural data",["Lakehouse","Catalog"]),Layer("Intelligence","Models and analytics",["ML","Decision support"]),Layer("Delivery","Operational use",["APIs","Dashboards"])]
architecture_summary=lambda layers=None:[vars(x) for x in (layers or reference_layers())]
completeness_report=lambda df:df.notna().mean().to_dict()
def consistency_checks(df): return {"nonnegative_area":bool((df["area"].dropna()>=0).all()),"nonnegative_production":bool((df["production"].dropna()>=0).all()),"gender_total":bool(((df["male"]+df["female"])==df["total"]).all())}
def quality_score(df,columns): return float(df[columns].notna().mean().mean())
range_check=lambda x,low,high:((np.asarray(x,dtype=float)>=low)&(np.asarray(x,dtype=float)<=high))
outlier_zscore=lambda x,threshold=3:np.abs((np.asarray(x)-np.mean(x))/(np.std(x) or 1))>threshold
select_deployment_mode=lambda connectivity,data_sensitivity,cloud_readiness:"hybrid" if data_sensitivity=="high" or connectivity=="limited" else "cloud"
def food_security_risk_score(price_change,rainfall_anomaly,production_change):
    return np.clip(.4*np.asarray(price_change)+.3*np.maximum(-np.asarray(rainfall_anomaly),0)+.3*np.maximum(-np.asarray(production_change),0),0,1)
def classify_risk(x): return "high" if x>=.5 else ("medium" if x>=.25 else "low")
standardized_precipitation_index=lambda x:(np.asarray(x)-np.mean(x))/(np.std(x) or 1)
rainfall_anomaly_percent=lambda x,c:100*(np.asarray(x)-np.asarray(c))/np.asarray(c)
cumulative_rainfall=lambda x:np.cumsum(x)
def moving_average_forecast(x,horizon,window=3): return np.repeat(np.mean(np.asarray(x)[-window:]),horizon)
def seasonal_naive(x,horizon,season=12): return np.resize(np.asarray(x)[-season:],horizon)
def trend_forecast(x,horizon):
    x=np.asarray(x); coef=np.polyfit(np.arange(len(x)),x,1); return np.polyval(coef,np.arange(len(x),len(x)+horizon))
def ensemble_forecast(forecasts,weights=None):
    a=np.asarray(forecasts,float); return np.average(a,axis=0,weights=weights)
def forecast_interval(point,residuals,confidence=.95):
    d=1.96*np.std(residuals); p=np.asarray(point); return p-d,p+d
mae=lambda y,p:float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
__getattr__ = dynamic_attribute
