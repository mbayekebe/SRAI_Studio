"""Multi-agent coordination utilities."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Message:
    sender:str
    recipient:str
    content:str
    topic:str="general"

class MessageBus:
    def __init__(self):
        self.messages=[]
    def send(self,message):
        self.messages.append(message)
    def inbox(self,recipient):
        return [m for m in self.messages if m.recipient==recipient]

def delegate_tasks(tasks,agents):
    assignments={}
    for i,task in enumerate(tasks):
        agent=agents[i%len(agents)]
        assignments.setdefault(agent,[]).append(task)
    return assignments

def consensus_vote(proposals):
    counts={}
    for proposal in proposals:
        counts[proposal]=counts.get(proposal,0)+1
    winner=max(counts,key=counts.get)
    return {"winner":winner,"votes":counts[winner],"counts":counts}

def aggregate_reports(reports):
    return {
        "agents":list(reports.keys()),
        "combined":"\n".join(f"{agent}: {report}" for agent,report in reports.items()),
    }
