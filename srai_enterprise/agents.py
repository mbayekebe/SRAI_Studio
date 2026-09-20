"""Enterprise multi-agent orchestration."""
from __future__ import annotations

class EnterpriseAgent:
    def __init__(self,name,capabilities):
        self.name=name
        self.capabilities=set(capabilities)

    def can_handle(self,task):
        return task in self.capabilities

def assign_tasks(tasks,agents):
    assignments=[]
    for task in tasks:
        eligible=[agent.name for agent in agents if agent.can_handle(task)]
        assignments.append({"task":task,"agents":eligible})
    return assignments

def supervisor_decision(agent_outputs,quality_key="score"):
    ranked=sorted(agent_outputs,key=lambda x:x.get(quality_key,0),reverse=True)
    return ranked[0] if ranked else None

def shared_memory_update(memory,key,value,agent):
    memory[key]={"value":value,"updated_by":agent}
    return memory

def agent_governance_check(agent,allowed_tools):
    unauthorized=agent.capabilities-set(allowed_tools)
    return {"approved":not unauthorized,"unauthorized":sorted(unauthorized)}
