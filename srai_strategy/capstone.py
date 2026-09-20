"""SRAI transformation capstone helpers."""
from __future__ import annotations
import numpy as np

def transformation_scorecard(strategy,foundations,delivery,adoption,governance,public_value):
    values=[strategy,foundations,delivery,adoption,governance,public_value]
    return {
        "strategy":float(strategy),
        "foundations":float(foundations),
        "delivery":float(delivery),
        "adoption":float(adoption),
        "governance":float(governance),
        "public_value":float(public_value),
        "overall":float(np.mean(values)),
    }

def final_readiness(checks):
    failed=[name for name,value in checks.items() if not value]
    return {
        "ready":not failed,
        "failed_checks":failed,
        "completion_rate":(len(checks)-len(failed))/max(len(checks),1),
    }

def srai_architecture():
    return [
        "Foundations and mathematics",
        "Statistics and inference",
        "Machine learning",
        "Generative AI",
        "MLOps and AI engineering",
        "Decision intelligence",
        "Applied AI studio",
        "Enterprise AI systems",
        "AI transformation and leadership",
    ]
