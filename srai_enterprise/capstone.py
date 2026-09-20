"""Enterprise AI Systems capstone helpers."""
from __future__ import annotations
import numpy as np

def enterprise_ai_scorecard(architecture,data,security,operations,governance):
    values=[architecture,data,security,operations,governance]
    return {
        "architecture":float(architecture),
        "data":float(data),
        "security":float(security),
        "operations":float(operations),
        "governance":float(governance),
        "overall":float(np.mean(values)),
    }

def volume8_readiness(checks):
    failed=[k for k,v in checks.items() if not v]
    return {
        "ready":not failed,
        "failed_checks":failed,
        "completion_rate":(len(checks)-len(failed))/max(len(checks),1),
    }
