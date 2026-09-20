"""Responsible AI leadership helpers."""
from __future__ import annotations

def leadership_commitments():
    return [
        "human_accountability",
        "fairness_and_inclusion",
        "privacy_and_security",
        "transparency",
        "contestability",
        "continuous_monitoring",
    ]

def executive_risk_acceptance(risk_level,controls_effective,owner_assigned):
    if risk_level in {"critical","high"} and not controls_effective:
        return {"accepted":False,"reason":"controls_not_effective"}
    if not owner_assigned:
        return {"accepted":False,"reason":"no_accountable_owner"}
    return {"accepted":True,"reason":"approved_with_accountability"}

def ethics_review(benefit,harm,fairness,transparency):
    score=.35*benefit-.30*harm+.20*fairness+.15*transparency
    return {"score":float(score),"approved":score>=.5}
