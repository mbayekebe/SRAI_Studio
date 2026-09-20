"""Decision framing utilities."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class DecisionProblem:
    objective:str
    alternatives:list
    criteria:list
    stakeholders:list=field(default_factory=list)
    constraints:list=field(default_factory=list)
    uncertainties:list=field(default_factory=list)

    def summary(self):
        return {
            "objective":self.objective,
            "alternatives":len(self.alternatives),
            "criteria":len(self.criteria),
            "stakeholders":len(self.stakeholders),
            "constraints":len(self.constraints),
            "uncertainties":len(self.uncertainties),
        }

def decision_statement(context,decision_owner,decision_deadline):
    return {
        "context":context,
        "decision_owner":decision_owner,
        "decision_deadline":decision_deadline,
    }

def stakeholder_map(stakeholders):
    return sorted(
        stakeholders,
        key=lambda x:(x.get("power",0),x.get("interest",0)),
        reverse=True,
    )

def constraint_check(alternative,constraints):
    failures=[]
    for constraint in constraints:
        if not constraint["predicate"](alternative):
            failures.append(constraint["name"])
    return {"feasible":not failures,"failures":failures}
