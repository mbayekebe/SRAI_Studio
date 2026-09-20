"""Planning and memory utilities for agents."""
from __future__ import annotations
from collections import deque

def decompose_goal(goal,subtasks):
    return [{"goal":goal,"step":i+1,"task":task,"status":"pending"} for i,task in enumerate(subtasks)]

def update_plan(plan,step,status,result=None):
    plan[step]["status"]=status
    if result is not None:
        plan[step]["result"]=result
    return plan

class ShortTermMemory:
    def __init__(self,capacity=10):
        self.items=deque(maxlen=capacity)
    def add(self,item):
        self.items.append(item)
    def read(self):
        return list(self.items)

class LongTermMemory:
    def __init__(self):
        self.records=[]
    def add(self,text,metadata=None):
        self.records.append({"text":text,"metadata":metadata or {}})
    def search(self,query,k=3):
        q=set(str(query).lower().split())
        scored=[]
        for i,record in enumerate(self.records):
            terms=set(record["text"].lower().split())
            score=len(q&terms)/max(len(q),1)
            scored.append((i,score,record))
        scored.sort(key=lambda x:x[1],reverse=True)
        return [{"index":i,"score":score,**record} for i,score,record in scored[:k]]

def summarize_memory(items,max_items=5):
    selected=list(items)[-max_items:]
    return " | ".join(str(item) for item in selected)
