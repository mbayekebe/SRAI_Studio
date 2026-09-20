"""National Agricultural Intelligence Platform architecture helpers."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PlatformLayer:
    name:str
    purpose:str
    components:list

def reference_layers():
    return [
        PlatformLayer("Sources","Operational and statistical data acquisition",[
            "Agricultural census","Annual surveys","Market prices","Weather",
            "Remote sensing","Administrative systems","Livestock systems"
        ]),
        PlatformLayer("Ingestion","Batch and streaming acquisition",[
            "APIs","Files","Databases","Mobile collection","Satellite feeds"
        ]),
        PlatformLayer("Lakehouse","Bronze, silver, and gold data zones",[
            "Raw storage","Validated data","Curated marts","Metadata catalog"
        ]),
        PlatformLayer("Analytics","Descriptive, predictive, and prescriptive analytics",[
            "Statistics","Forecasting","Risk models","Optimization","Simulation"
        ]),
        PlatformLayer("AI Services","Generative and agentic capabilities",[
            "RAG","Forecast assistants","Survey copilot","Policy assistant"
        ]),
        PlatformLayer("Experience","Decision interfaces",[
            "Dashboards","Alerts","Reports","APIs","Mobile applications"
        ]),
        PlatformLayer("Governance","Cross-cutting controls",[
            "Security","Quality","Lineage","Privacy","Model governance","Audit"
        ]),
    ]

def architecture_summary():
    return {
        "source_domains":7,
        "core_layers":len(reference_layers()),
        "governance_cross_cutting":True,
        "primary_users":[
            "Ministry leadership","National statistics office","Extension services",
            "Researchers","Development partners","Regional offices"
        ],
    }

def select_deployment_mode(connectivity,data_sensitivity,cloud_readiness):
    if data_sensitivity=="high" and cloud_readiness=="low":
        return "on_premises"
    if connectivity=="limited":
        return "hybrid_edge"
    if cloud_readiness=="high":
        return "cloud_first"
    return "hybrid"
