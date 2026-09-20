"""Executive AI dashboard helpers."""
from __future__ import annotations
import pandas as pd

def executive_kpis(portfolio_value,adoption,risk_exposure,delivery_rate,talent_coverage):
    return {
        "portfolio_value":float(portfolio_value),
        "adoption":float(adoption),
        "risk_exposure":float(risk_exposure),
        "delivery_rate":float(delivery_rate),
        "talent_coverage":float(talent_coverage),
    }

def executive_signal(value,green,amber,higher_is_better=True):
    if higher_is_better:
        return "green" if value>=green else "amber" if value>=amber else "red"
    return "green" if value<=green else "amber" if value<=amber else "red"

def initiative_register(rows):
    return pd.DataFrame(rows).sort_values(["status","priority"],ascending=[True,False])

def board_readiness(strategy,portfolio,risk,people,governance):
    checks={"strategy":strategy,"portfolio":portfolio,"risk":risk,"people":people,"governance":governance}
    return {"checks":checks,"ready":all(checks.values())}
