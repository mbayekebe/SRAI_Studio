"""Agricultural data ingestion helpers."""
from __future__ import annotations
import pandas as pd
import numpy as np

def standardize_columns(df):
    out=df.copy()
    out.columns=[str(c).strip().lower().replace(" ","_") for c in out.columns]
    return out

def normalize_crop_names(series):
    mapping={
        "maize":"maize","corn":"maize","zea mays":"maize",
        "teff":"teff","wheat":"wheat","sorghum":"sorghum",
        "groundnut":"groundnut","peanut":"groundnut",
    }
    return series.astype(str).str.strip().str.lower().map(lambda x:mapping.get(x,x))

def ingest_csv_like(records):
    df=pd.DataFrame(records)
    df=standardize_columns(df)
    if "crop" in df.columns:
        df["crop"]=normalize_crop_names(df["crop"])
    return df

def bronze_silver_gold(df):
    bronze=df.copy()
    silver=bronze.drop_duplicates().copy()
    numeric_cols=silver.select_dtypes(include=[np.number]).columns
    silver[numeric_cols]=silver[numeric_cols].replace([np.inf,-np.inf],np.nan)
    gold=silver.copy()
    return {"bronze":bronze,"silver":silver,"gold":gold}

def ingestion_manifest(source_name,row_count,columns,load_type="batch"):
    return {
        "source_name":source_name,
        "row_count":int(row_count),
        "columns":list(columns),
        "load_type":load_type,
    }
