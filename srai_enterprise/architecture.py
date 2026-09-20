"""Enterprise AI architecture helpers."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ArchitectureLayer:
    name:str
    purpose:str
    components:list
    criticality:str="medium"

def enterprise_ai_reference_architecture():
    return [
        ArchitectureLayer("Channels","User and system interaction",["Web","Mobile","APIs","Copilots"],"medium"),
        ArchitectureLayer("Experience","Dashboards, workflows, and assistants",["BI","Chat","Alerts","Case management"],"high"),
        ArchitectureLayer("AI Services","Predictive and generative services",["Models","RAG","Agents","Optimization"],"high"),
        ArchitectureLayer("Orchestration","Policies, routing, tools, and workflows",["Model gateway","Tool router","Workflow engine"],"high"),
        ArchitectureLayer("Data Products","Governed analytical products",["Lakehouse","Data marts","Feature store","Vector store"],"high"),
        ArchitectureLayer("Integration","Enterprise connectivity",["APIs","Events","ETL","CDC"],"high"),
        ArchitectureLayer("Platforms","Compute and runtime",["Containers","Kubernetes","Cloud","On-premises"],"high"),
        ArchitectureLayer("Trust","Cross-cutting controls",["IAM","Security","Privacy","Governance","Observability"],"critical"),
    ]

def deployment_pattern(sensitivity,latency,scale,cloud_readiness):
    if sensitivity=="high" and cloud_readiness=="low":
        return "private_on_premises"
    if latency=="ultra_low":
        return "edge_or_local"
    if scale=="high" and cloud_readiness=="high":
        return "cloud_native"
    return "hybrid"

def architecture_score(availability,security,scalability,interoperability,governance):
    values=[availability,security,scalability,interoperability,governance]
    return {
        "availability":float(availability),
        "security":float(security),
        "scalability":float(scalability),
        "interoperability":float(interoperability),
        "governance":float(governance),
        "overall":sum(values)/len(values),
    }
