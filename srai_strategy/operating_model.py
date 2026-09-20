"""AI operating-model helpers."""
from __future__ import annotations

def operating_model_pattern(central_control,domain_autonomy,shared_platform):
    if central_control=="high" and domain_autonomy=="low":
        return "centralized"
    if domain_autonomy=="high" and shared_platform:
        return "federated"
    if domain_autonomy=="high":
        return "decentralized"
    return "hub_and_spoke"

def raci_matrix(activities,roles,assignments):
    rows=[]
    for activity in activities:
        row={"activity":activity}
        for role in roles:
            row[role]=assignments.get(activity,{}).get(role,"")
        rows.append(row)
    return rows

def governance_forum(name,cadence,members,decision_rights):
    return {
        "name":name,
        "cadence":cadence,
        "members":list(members),
        "decision_rights":list(decision_rights),
    }

def team_topology():
    return {
        "AI_Centre_of_Excellence":["standards","platform","governance","enablement"],
        "Domain_AI_Teams":["use_cases","data_products","adoption"],
        "Risk_and_Assurance":["validation","audit","compliance"],
    }
