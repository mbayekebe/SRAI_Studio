"""Agent foundations."""
from __future__ import annotations
from dataclasses import dataclass,field

@dataclass
class AgentState:
    goal:str
    observations:list=field(default_factory=list)
    actions:list=field(default_factory=list)
    completed:bool=False

class RuleBasedAgent:
    def __init__(self,rules):
        self.rules=list(rules)
    def choose_action(self,state):
        for predicate,action in self.rules:
            if predicate(state):
                return action(state)
        return {"type":"stop","reason":"No applicable rule"}

def react_step(thought,action,observation):
    return {
        "thought":thought,
        "action":action,
        "observation":observation,
    }

def run_agent(agent,state,environment,max_steps=10):
    trace=[]
    for _ in range(max_steps):
        action=agent.choose_action(state)
        observation=environment(action,state)
        state.actions.append(action)
        state.observations.append(observation)
        trace.append(react_step("Select next useful action",action,observation))
        if action.get("type")=="stop" or observation.get("done"):
            state.completed=True
            break
    return state,trace
