"""AI workforce and change-management helpers."""
from __future__ import annotations
import pandas as pd

def skills_gap(current,target):
    rows=[]
    for skill,target_level in target.items():
        current_level=current.get(skill,0)
        rows.append({"skill":skill,"current":current_level,"target":target_level,"gap":max(0,target_level-current_level)})
    return pd.DataFrame(rows).sort_values("gap",ascending=False)

def learning_path(role):
    paths={
        "executive":["AI strategy","responsible AI","value realization"],
        "manager":["use-case discovery","change leadership","decision intelligence"],
        "technical":["data engineering","ML","MLOps","enterprise AI"],
        "oversight":["risk","audit","governance","security"],
    }
    return paths.get(role,["AI literacy"])

def change_readiness(leadership,communication,skills,incentives,participation):
    checks={
        "leadership":leadership,
        "communication":communication,
        "skills":skills,
        "incentives":incentives,
        "participation":participation,
    }
    return {"checks":checks,"score":sum(bool(v) for v in checks.values())/len(checks)}
