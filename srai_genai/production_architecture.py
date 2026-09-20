"""Production LLM architecture helpers."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Component:
    name:str
    responsibility:str
    criticality:str="medium"

def reference_architecture():
    return [
        Component("API Gateway","Authentication, quotas, routing","high"),
        Component("Orchestrator","Prompting, tools, retrieval, policies","high"),
        Component("Retriever","Document and vector search","high"),
        Component("Model Gateway","Model abstraction, fallback, load balancing","high"),
        Component("Guardrails","Input and output safety controls","high"),
        Component("Observability","Tracing, cost, quality, incidents","high"),
        Component("Cache","Response and embedding caching","medium"),
        Component("Human Review","Escalation for high-impact cases","high"),
    ]

def routing_decision(task_complexity,sensitivity,latency_priority):
    if sensitivity=="high":
        return "private_model"
    if latency_priority=="high" and task_complexity=="low":
        return "small_fast_model"
    if task_complexity=="high":
        return "large_reasoning_model"
    return "general_model"

def resilience_plan():
    return {
        "timeouts":True,
        "retries_with_backoff":True,
        "model_fallback":True,
        "retrieval_fallback":True,
        "circuit_breaker":True,
        "manual_escalation":True,
    }

def capacity_estimate(requests_per_second,average_latency_seconds,headroom=.3):
    concurrent=requests_per_second*average_latency_seconds
    return {
        "base_concurrency":float(concurrent),
        "recommended_concurrency":int(np.ceil(concurrent*(1+headroom))),
    }
