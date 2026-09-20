"""Grounding and citation utilities."""
from __future__ import annotations
import re

def extract_claims(answer):
    sentences=re.split(r"(?<=[.!?])\s+",str(answer).strip())
    return [s for s in sentences if s]

def citation_coverage(answer):
    claims=extract_claims(answer)
    cited=[bool(re.search(r"\[(?:Source|S)\s*\d+",claim,re.I)) for claim in claims]
    return float(sum(cited)/len(cited)) if cited else 0.0

def supported_sentence(sentence,contexts):
    terms=set(re.findall(r"\b\w+\b",str(sentence).lower()))
    if not terms:
        return False
    best=0
    for context in contexts:
        cterms=set(re.findall(r"\b\w+\b",str(context).lower()))
        best=max(best,len(terms&cterms)/len(terms))
    return best>=0.5

def hallucination_flags(answer,contexts):
    return [
        {"claim":claim,"supported":supported_sentence(claim,contexts)}
        for claim in extract_claims(answer)
    ]

def format_citations(retrieved):
    return [
        f"[Source {item['source_id']}, Chunk {item['chunk_id']}]"
        for item in retrieved
    ]

def answerability_score(query,retrieved):
    query_terms=set(re.findall(r"\b\w+\b",str(query).lower()))
    if not query_terms or not retrieved:
        return 0.0
    scores=[]
    for item in retrieved:
        terms=set(re.findall(r"\b\w+\b",item["text"].lower()))
        scores.append(len(query_terms&terms)/len(query_terms))
    return float(max(scores))
