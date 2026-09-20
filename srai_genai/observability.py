"""LLM observability and cost management utilities."""
from __future__ import annotations
import numpy as np

def estimate_token_cost(input_tokens,output_tokens,input_rate_per_million,output_rate_per_million):
    return float(
        input_tokens/1_000_000*input_rate_per_million+
        output_tokens/1_000_000*output_rate_per_million
    )

def latency_summary(latencies_ms):
    x=np.asarray(latencies_ms,float)
    return {
        "mean_ms":float(np.mean(x)),
        "p50_ms":float(np.quantile(x,.5)),
        "p95_ms":float(np.quantile(x,.95)),
        "p99_ms":float(np.quantile(x,.99)),
    }

def request_trace(request_id,model,prompt_tokens,completion_tokens,latency_ms,status="ok",metadata=None):
    return {
        "request_id":request_id,
        "model":model,
        "prompt_tokens":int(prompt_tokens),
        "completion_tokens":int(completion_tokens),
        "latency_ms":float(latency_ms),
        "status":status,
        "metadata":metadata or {},
    }

def aggregate_traces(traces):
    if not traces: return {}
    return {
        "requests":len(traces),
        "total_prompt_tokens":sum(t["prompt_tokens"] for t in traces),
        "total_completion_tokens":sum(t["completion_tokens"] for t in traces),
        "error_rate":sum(t["status"]!="ok" for t in traces)/len(traces),
        "mean_latency_ms":float(np.mean([t["latency_ms"] for t in traces])),
    }

def budget_alert(current_cost,budget,warning_fraction=.8):
    if current_cost>=budget: return "critical"
    if current_cost>=warning_fraction*budget: return "warning"
    return "normal"
