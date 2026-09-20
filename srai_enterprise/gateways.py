"""API and model gateway helpers."""
from __future__ import annotations

def route_model(task_type,sensitivity,latency_priority):
    if sensitivity=="high":
        return "private-model"
    if task_type=="reasoning":
        return "reasoning-model"
    if latency_priority=="high":
        return "small-fast-model"
    return "general-model"

def rate_limit(requests,limit):
    return {"allowed":requests<limit,"remaining":max(limit-requests,0)}

def circuit_breaker(error_rate,threshold=.2):
    return "open" if error_rate>=threshold else "closed"

def gateway_policy(user,model,allowed_models):
    return {
        "allowed":model in allowed_models.get(user.get("role"),[]),
        "role":user.get("role"),
        "model":model,
    }

def model_fallback(primary_status,fallbacks):
    if primary_status=="healthy":
        return "primary"
    return fallbacks[0] if fallbacks else None
