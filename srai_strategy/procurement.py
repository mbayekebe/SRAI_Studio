"""AI procurement and partnership helpers."""
from __future__ import annotations
import pandas as pd

def vendor_score(functional_fit,security,interoperability,total_cost,capacity_transfer):
    return float(.25*functional_fit+.25*security+.20*interoperability+.15*(1-total_cost)+.15*capacity_transfer)

def rank_vendors(vendors):
    rows=[]
    for vendor in vendors:
        rows.append({**vendor,"score":vendor_score(
            vendor["functional_fit"],vendor["security"],vendor["interoperability"],
            vendor["total_cost"],vendor["capacity_transfer"],
        )})
    return pd.DataFrame(rows).sort_values("score",ascending=False)

def contract_clauses():
    return [
        "data_ownership",
        "security_and_privacy",
        "model_transparency",
        "audit_rights",
        "service_levels",
        "portability_and_exit",
        "capacity_transfer",
    ]

def partnership_model(public_value,commercial_incentive,capacity_transfer):
    if public_value=="high" and capacity_transfer=="high":
        return "strategic_public_private_partnership"
    if commercial_incentive=="low":
        return "development_partnership"
    return "managed_vendor_relationship"
