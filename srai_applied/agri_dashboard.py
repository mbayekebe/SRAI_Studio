"""Executive dashboard helpers."""
from __future__ import annotations
import pandas as pd
import numpy as np

def kpi_summary(df):
    return {
        "total_area":float(df["area"].sum()) if "area" in df else 0.0,
        "total_production":float(df["production"].sum()) if "production" in df else 0.0,
        "average_yield":float(df["yield_per_hectare"].mean()) if "yield_per_hectare" in df else 0.0,
        "regions_covered":int(df["region"].nunique()) if "region" in df else 0,
        "crops_covered":int(df["crop"].nunique()) if "crop" in df else 0,
    }

def alert_table(df,risk_col="risk_score"):
    out=df.copy()
    out["risk_level"]=out[risk_col].map(
        lambda s:"critical" if s>=.75 else "high" if s>=.5 else "medium" if s>=.25 else "low"
    )
    return out.sort_values(risk_col,ascending=False)

def dashboard_readiness(kpis,quality_score,refresh_age_hours):
    checks={
        "kpis_available":bool(kpis),
        "quality_acceptable":quality_score>=.8,
        "fresh_data":refresh_age_hours<=24,
    }
    return {
        "checks":checks,
        "ready":all(checks.values()),
    }
