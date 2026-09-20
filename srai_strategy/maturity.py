"""AI maturity assessment helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd

DIMENSIONS=["strategy","data","technology","talent","governance","delivery"]

def maturity_level(score):
    if score>=.85: return "optimized"
    if score>=.65: return "scaled"
    if score>=.45: return "managed"
    if score>=.25: return "emerging"
    return "initial"

def assess_maturity(scores):
    values={dimension:float(scores.get(dimension,0)) for dimension in DIMENSIONS}
    overall=float(np.mean(list(values.values())))
    return {"dimensions":values,"overall":overall,"level":maturity_level(overall)}

def maturity_gaps(current,target):
    rows=[]
    for dimension in DIMENSIONS:
        gap=max(0,float(target.get(dimension,0)-current.get(dimension,0)))
        rows.append({"dimension":dimension,"current":current.get(dimension,0),"target":target.get(dimension,0),"gap":gap})
    return pd.DataFrame(rows).sort_values("gap",ascending=False)

def next_best_capabilities(gap_frame,n=3):
    return gap_frame.head(n)["dimension"].tolist()
