"""Event-driven AI helpers."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import time

@dataclass
class Event:
    event_type:str
    entity_id:str
    payload:dict
    timestamp:float|None=None

    def to_dict(self):
        row=asdict(self)
        row["timestamp"]=self.timestamp if self.timestamp is not None else time.time()
        return row

class EventBus:
    def __init__(self):
        self.subscribers={}
        self.history=[]

    def subscribe(self,event_type,handler):
        self.subscribers.setdefault(event_type,[]).append(handler)
        return self

    def publish(self,event):
        row=event.to_dict()
        self.history.append(row)
        results=[]
        for handler in self.subscribers.get(event.event_type,[]):
            results.append(handler(row))
        return results

def idempotency_key(event):
    return f"{event.event_type}:{event.entity_id}:{event.payload.get('version','0')}"

def event_lag_seconds(event_timestamp,current_timestamp):
    return max(0.0,float(current_timestamp-event_timestamp))
