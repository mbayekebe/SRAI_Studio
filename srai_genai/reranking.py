"""Retrieval reranking utilities."""
from __future__ import annotations
import re
import numpy as np

def lexical_overlap_score(query,document):
    q=set(re.findall(r"\b\w+\b",str(query).lower()))
    d=set(re.findall(r"\b\w+\b",str(document).lower()))
    return float(len(q&d)/max(len(q),1))

def length_penalty(document,ideal_words=80):
    n=len(str(document).split())
    return float(np.exp(-abs(n-ideal_words)/max(ideal_words,1)))

def rerank(query,candidates,semantic_weight=.7,lexical_weight=.3):
    rows=[]
    for candidate in candidates:
        lexical=lexical_overlap_score(query,candidate["text"])
        semantic=float(candidate.get("score",0.0))
        score=semantic_weight*semantic+lexical_weight*lexical
        row=dict(candidate)
        row.update({"lexical_score":lexical,"rerank_score":score})
        rows.append(row)
    return sorted(rows,key=lambda x:x["rerank_score"],reverse=True)

def reciprocal_rank_fusion_results(rankings,k=60):
    scores={}
    records={}
    for ranking in rankings:
        for rank,item in enumerate(ranking,1):
            key=item.get("chunk_id",item.get("index"))
            scores[key]=scores.get(key,0)+1/(k+rank)
            records[key]=item
    return [
        {**records[key],"rrf_score":score}
        for key,score in sorted(scores.items(),key=lambda x:x[1],reverse=True)
    ]
