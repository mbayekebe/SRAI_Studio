"""Enterprise RAG architecture helpers."""
from __future__ import annotations
import re

def classify_document(text,classification_rules=None):
    rules=classification_rules or {
        "restricted":["confidential","salary","personal data"],
        "internal":["internal use","draft"],
    }
    lower=str(text).lower()
    for label,terms in rules.items():
        if any(term in lower for term in terms):
            return label
    return "public"

def retrieval_scope(user,documents):
    allowed=[]
    clearance=user.get("clearance","public")
    ranking={"public":0,"internal":1,"restricted":2}
    for document in documents:
        if ranking[document.get("classification","public")]<=ranking[clearance]:
            allowed.append(document)
    return allowed

def citation_response(answer,retrieved):
    citations=[
        f"[{item.get('source','unknown')}#{item.get('chunk_id',0)}]"
        for item in retrieved
    ]
    return {"answer":answer,"citations":citations}

def rag_readiness(indexed,permissions,quality_monitoring,audit_logging):
    checks={
        "indexed":bool(indexed),
        "permissions":bool(permissions),
        "quality_monitoring":bool(quality_monitoring),
        "audit_logging":bool(audit_logging),
    }
    return {"checks":checks,"ready":all(checks.values())}
