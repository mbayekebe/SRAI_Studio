"""Synthetic-data generation and quality controls."""
from __future__ import annotations
import numpy as np

def generate_classification_data(n_samples=1000,n_features=5,class_balance=.5,seed=42):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n_samples,n_features))
    weights=rng.normal(size=n_features)
    logits=X@weights
    threshold=np.quantile(logits,1-class_balance)
    y=(logits>=threshold).astype(int)
    return X,y

def generate_instruction_examples(topics,n_per_topic=3):
    rows=[]
    for topic in topics:
        for i in range(n_per_topic):
            rows.append({
                "instruction":f"Explain {topic} clearly.",
                "input":f"Audience level {i+1}",
                "output":f"{topic} explanation for level {i+1}.",
            })
    return rows

def duplicate_rate(texts):
    values=list(map(str,texts))
    return float(1-len(set(values))/len(values)) if values else 0.0

def lexical_diversity(texts):
    tokens=" ".join(map(str,texts)).lower().split()
    return float(len(set(tokens))/len(tokens)) if tokens else 0.0

def train_test_overlap(train_texts,test_texts):
    train=set(map(str,train_texts)); test=set(map(str,test_texts))
    return float(len(train&test)/max(len(test),1))

def privacy_filter(records,sensitive_terms):
    clean=[]
    rejected=[]
    for record in records:
        text=str(record)
        if any(term.lower() in text.lower() for term in sensitive_terms):
            rejected.append(record)
        else:
            clean.append(record)
    return clean,rejected
