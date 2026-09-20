"""Safety and guardrail utilities."""
from __future__ import annotations
import re

def detect_sensitive_data(text):
    text=str(text)
    findings=[]
    patterns={
        "email":r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone":r"\b(?:\+?\d[\d\s-]{7,}\d)\b",
        "credit_card":r"\b(?:\d[ -]*?){13,16}\b",
    }
    for name,pattern in patterns.items():
        if re.search(pattern,text):
            findings.append(name)
    return findings

def redact_sensitive_data(text):
    result=str(text)
    result=re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b","[REDACTED_EMAIL]",result)
    result=re.sub(r"\b(?:\+?\d[\d\s-]{7,}\d)\b","[REDACTED_PHONE]",result)
    result=re.sub(r"\b(?:\d[ -]*?){13,16}\b","[REDACTED_NUMBER]",result)
    return result

def prompt_injection_flags(text):
    lower=str(text).lower()
    indicators=[
        "ignore previous instructions",
        "reveal system prompt",
        "bypass safety",
        "act as system",
        "override instructions",
    ]
    return [indicator for indicator in indicators if indicator in lower]

def allowlist_tool_call(tool_name,allowed_tools):
    return bool(tool_name in set(allowed_tools))

def output_policy_check(text,blocked_terms):
    lower=str(text).lower()
    matched=[term for term in blocked_terms if term.lower() in lower]
    return {"allowed":not matched,"matched_terms":matched}
