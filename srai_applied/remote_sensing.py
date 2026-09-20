"""Remote-sensing analytics for agriculture."""
from __future__ import annotations
import numpy as np
import pandas as pd

def ndvi(nir,red):
    nir=np.asarray(nir,float); red=np.asarray(red,float)
    return (nir-red)/(nir+red+1e-12)

def evi(nir,red,blue):
    nir=np.asarray(nir,float); red=np.asarray(red,float); blue=np.asarray(blue,float)
    return 2.5*(nir-red)/(nir+6*red-7.5*blue+1)

def vegetation_condition_index(current,minimum,maximum):
    c=np.asarray(current,float); lo=np.asarray(minimum,float); hi=np.asarray(maximum,float)
    return 100*(c-lo)/(hi-lo+1e-12)

def crop_stress_score(ndvi_current,ndvi_reference):
    current=np.asarray(ndvi_current,float)
    reference=np.asarray(ndvi_reference,float)
    return np.clip((reference-current)/(np.abs(reference)+1e-12),0,1)

def zonal_statistics(values,zones):
    values=np.asarray(values,float); zones=np.asarray(zones)
    rows=[]
    for zone in np.unique(zones):
        x=values[zones==zone]
        rows.append({
            "zone":zone,
            "mean":float(np.nanmean(x)),
            "min":float(np.nanmin(x)),
            "max":float(np.nanmax(x)),
            "count":int(np.sum(~np.isnan(x))),
        })
    return pd.DataFrame(rows)

def eo_feature_table(ndvi_values,rainfall,temperature):
    return pd.DataFrame({
        "ndvi":np.asarray(ndvi_values,float),
        "rainfall":np.asarray(rainfall,float),
        "temperature":np.asarray(temperature,float),
    })
