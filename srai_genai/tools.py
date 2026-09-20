"""Tool and function calling primitives."""
from __future__ import annotations
import json
from dataclasses import dataclass

@dataclass
class ToolSpec:
    name:str
    description:str
    parameters:dict

class ToolRegistry:
    def __init__(self):
        self._tools={}
        self._specs={}
    def register(self,spec,function):
        self._specs[spec.name]=spec
        self._tools[spec.name]=function
        return self
    def list_specs(self):
        return list(self._specs.values())
    def execute(self,name,arguments):
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](**arguments)

def validate_arguments(spec,arguments):
    required=spec.parameters.get("required",[])
    properties=spec.parameters.get("properties",{})
    missing=[name for name in required if name not in arguments]
    unknown=[name for name in arguments if name not in properties]
    return {"valid":not missing and not unknown,"missing":missing,"unknown":unknown}

def parse_tool_call(payload):
    if isinstance(payload,str):
        payload=json.loads(payload)
    if "name" not in payload or "arguments" not in payload:
        raise ValueError("Tool call requires name and arguments.")
    return payload["name"],payload["arguments"]

def tool_result_message(name,result):
    return {"role":"tool","name":name,"content":json.dumps(result,default=str)}
