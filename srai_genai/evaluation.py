"""LLM evaluation utilities."""
from __future__ import annotations
import re
import numpy as np

def exact_match(prediction,reference):
    return float(str(prediction).strip().lower()==str(reference).strip().lower())

def token_f1(prediction,reference):
    p=re.findall(r"\b\w+\b",str(prediction).lower())
    r=re.findall(r"\b\w+\b",str(reference).lower())
    if not p and not r: return 1.0
    if not p or not r: return 0.0
    common=0
    r_counts={}
    for token in r: r_counts[token]=r_counts.get(token,0)+1
    for token in p:
        if r_counts.get(token,0)>0:
            common+=1; r_counts[token]-=1
    precision=common/len(p); recall=common/len(r)
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.0

def groundedness_score(answer,contexts):
    terms=set(re.findall(r"\b\w+\b",str(answer).lower()))
    if not terms: return 0.0
    context_terms=set()
    for c in contexts:
        context_terms |= set(re.findall(r"\b\w+\b",str(c).lower()))
    return float(len(terms&context_terms)/len(terms))

def pairwise_preference_rate(scores_a,scores_b):
    a=np.asarray(scores_a,float); b=np.asarray(scores_b,float)
    return {
        "a_win_rate":float(np.mean(a>b)),
        "b_win_rate":float(np.mean(b>a)),
        "tie_rate":float(np.mean(a==b)),
    }

def aggregate_evaluation(rows):
    if not rows: return {}
    keys=rows[0].keys()
    return {k:float(np.mean([row[k] for row in rows])) for k in keys}
