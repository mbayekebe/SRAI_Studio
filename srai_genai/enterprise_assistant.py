"""Enterprise conversational AI helpers."""
from __future__ import annotations
from dataclasses import dataclass,field

@dataclass
class ConversationState:
    user_id:str
    messages:list=field(default_factory=list)
    permissions:set=field(default_factory=set)
    context:dict=field(default_factory=dict)

def add_message(state,role,content):
    state.messages.append({"role":role,"content":content})
    return state

def permission_check(state,required_permission):
    return bool(required_permission in state.permissions)

def route_intent(text):
    lower=str(text).lower()
    if any(word in lower for word in ["budget","revenue","expenditure"]):
        return "finance"
    if any(word in lower for word in ["farm","crop","livestock","census"]):
        return "agriculture"
    if any(word in lower for word in ["policy","governance","compliance"]):
        return "governance"
    return "general"

def escalation_required(confidence,sensitivity,policy_violation=False):
    return bool(policy_violation or sensitivity=="high" or confidence<.55)

def conversation_summary(messages,max_messages=6):
    selected=messages[-max_messages:]
    return " | ".join(f"{m['role']}: {m['content']}" for m in selected)
