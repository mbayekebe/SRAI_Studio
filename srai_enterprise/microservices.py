"""Enterprise AI microservice helpers."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Service:
    name:str
    responsibility:str
    dependencies:list
    replicas:int=1

def service_graph(services):
    return {service.name:list(service.dependencies) for service in services}

def deployment_units(services):
    return [
        {
            "service":s.name,
            "replicas":s.replicas,
            "dependencies":s.dependencies,
        }
        for s in services
    ]

def health_summary(statuses):
    unhealthy=[name for name,status in statuses.items() if status!="healthy"]
    return {"healthy":not unhealthy,"unhealthy_services":unhealthy}

def saga_result(steps):
    completed=[]
    for step in steps:
        if step["status"]=="failed":
            return {
                "status":"compensating",
                "completed":completed,
                "failed":step["name"],
            }
        completed.append(step["name"])
    return {"status":"completed","completed":completed}
