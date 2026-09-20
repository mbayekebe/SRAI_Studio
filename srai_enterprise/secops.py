"""AI security operations helpers."""
from __future__ import annotations

def threat_score(likelihood,impact,detectability):
    return float(likelihood*impact*(1-detectability))

def prompt_attack_flags(text):
    lower=str(text).lower()
    patterns=["ignore previous instructions","reveal system prompt","bypass policy","exfiltrate data"]
    return [p for p in patterns if p in lower]

def incident_severity(confidentiality,integrity,availability):
    score=confidentiality+integrity+availability
    return "critical" if score>=8 else "high" if score>=6 else "medium" if score>=3 else "low"

def secops_playbook(incident_type):
    playbooks={
        "prompt_injection":["block request","capture evidence","review logs","update filters"],
        "data_leak":["revoke access","rotate secrets","notify owner","investigate scope"],
        "model_abuse":["rate limit","suspend identity","review outputs","tighten policy"],
    }
    return playbooks.get(incident_type,["triage","contain","investigate","recover"])
