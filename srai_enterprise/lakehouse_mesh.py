"""Lakehouse and data-mesh utilities."""
from __future__ import annotations
import pandas as pd

def medallion_layers(raw_frame,validation_function=None,curation_function=None):
    bronze=raw_frame.copy()
    silver=validation_function(bronze.copy()) if validation_function else bronze.drop_duplicates()
    gold=curation_function(silver.copy()) if curation_function else silver.copy()
    return {"bronze":bronze,"silver":silver,"gold":gold}

def data_product_contract(name,owner,domain,schema,quality_sla,freshness_hours):
    return {
        "name":name,
        "owner":owner,
        "domain":domain,
        "schema":dict(schema),
        "quality_sla":float(quality_sla),
        "freshness_hours":float(freshness_hours),
    }

def contract_check(frame,contract):
    missing=[column for column in contract["schema"] if column not in frame.columns]
    quality=1-float(frame.isna().mean().mean()) if len(frame.columns) else 0
    return {
        "schema_valid":not missing,
        "missing_columns":missing,
        "quality":quality,
        "quality_sla_met":quality>=contract["quality_sla"],
    }

def domain_catalog(products):
    return pd.DataFrame(products).sort_values(["domain","name"])
