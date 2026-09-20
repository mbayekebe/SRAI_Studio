"""Optimization for decision support."""
from __future__ import annotations
import itertools
import numpy as np

def feasible_solutions(bounds,constraints):
    grids=[range(low,high+1) for low,high in bounds]
    solutions=[]
    for candidate in itertools.product(*grids):
        x=np.asarray(candidate,float)
        if all(constraint(x) for constraint in constraints):
            solutions.append(x)
    return solutions

def maximize_linear(objective,bounds,constraints):
    objective=np.asarray(objective,float)
    solutions=feasible_solutions(bounds,constraints)
    if not solutions:
        raise ValueError("No feasible solution.")
    values=[float(objective@x) for x in solutions]
    idx=int(np.argmax(values))
    return {"solution":solutions[idx],"objective":values[idx],"evaluated":len(solutions)}

def minimize_cost(costs,requirements,contributions,bounds):
    costs=np.asarray(costs,float)
    requirements=np.asarray(requirements,float)
    contributions=np.asarray(contributions,float)
    constraints=[
        lambda x,j=j: contributions[:,j]@x>=requirements[j]
        for j in range(len(requirements))
    ]
    solutions=feasible_solutions(bounds,constraints)
    values=[float(costs@x) for x in solutions]
    idx=int(np.argmin(values))
    return {"solution":solutions[idx],"cost":values[idx],"evaluated":len(solutions)}

def shadow_value(objective_with_relaxation,objective_baseline,relaxation):
    return float((objective_with_relaxation-objective_baseline)/relaxation)
