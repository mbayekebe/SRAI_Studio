import numpy as np
import pandas as pd
def coverage_rate(frame_count,enumerated_count):
    f=np.asarray(frame_count,float); e=np.asarray(enumerated_count,float)
    return np.divide(e,f,out=np.zeros_like(e),where=f>0)
def nonresponse_rate(selected,completed):
    s=np.asarray(selected,float); c=np.asarray(completed,float)
    return np.divide(s-c,s,out=np.zeros_like(s),where=s>0)
def structural_indicator_table(df):
    return {"holdings":int(len(df)),
            "total_area":float(df["area"].sum()) if "area" in df else 0.0,
            "mean_holding_size":float(df["area"].mean()) if "area" in df else 0.0,
            "female_managed_share":float(np.mean(df["manager_sex"].astype(str).str.lower()=="female")) if "manager_sex" in df else 0.0,
            "irrigated_share":float(np.mean(df["irrigated"].astype(bool))) if "irrigated" in df else 0.0}
def census_consistency_flags(df):
    flags=pd.DataFrame(index=df.index)
    if {"crop_area","total_area"}.issubset(df.columns): flags["crop_area_gt_total"]=df["crop_area"]>df["total_area"]
    if {"cattle","livestock_total"}.issubset(df.columns): flags["cattle_gt_total_livestock"]=df["cattle"]>df["livestock_total"]
    if {"members","workers"}.issubset(df.columns): flags["workers_gt_members"]=df["workers"]>df["members"]
    return flags
def regional_census_summary(df):
    return df.groupby("region",dropna=False).agg(holdings=("holding_id","nunique"),area=("area","sum"),livestock=("livestock_total","sum")).reset_index()
