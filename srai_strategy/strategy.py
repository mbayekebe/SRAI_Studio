"""AI strategy and transformation helpers."""
from __future__ import annotations
import numpy as np

def strategic_alignment_score(mission,stakeholders,data,technology,governance):
    values=[mission,stakeholders,data,technology,governance]
    return {
        "mission":float(mission),
        "stakeholders":float(stakeholders),
        "data":float(data),
        "technology":float(technology),
        "governance":float(governance),
        "overall":float(np.mean(values)),
    }

def strategic_pillars():
    return [
        "mission_and_outcomes",
        "data_and_digital_foundations",
        "responsible_ai",
        "talent_and_change",
        "delivery_and_operations",
        "partnerships_and_ecosystem",
    ]

def initiative_fit(value,feasibility,risk,strategic_alignment):
    return float(.35*value+.25*feasibility-.15*risk+.25*strategic_alignment)

def roadmap_phases():
    return ["Mobilize","Foundation","Pilot","Scale","Institutionalize"]
