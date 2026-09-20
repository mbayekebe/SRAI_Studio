"""Model Context Protocol concept helpers."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class MCPResource:
    uri:str
    name:str
    mime_type:str
    description:str=""

@dataclass
class MCPTool:
    name:str
    description:str
    input_schema:dict

@dataclass
class MCPPrompt:
    name:str
    description:str
    arguments:list

class MCPServerManifest:
    def __init__(self,name,version):
        self.name=name
        self.version=version
        self.resources=[]
        self.tools=[]
        self.prompts=[]
    def add_resource(self,resource):
        self.resources.append(resource); return self
    def add_tool(self,tool):
        self.tools.append(tool); return self
    def add_prompt(self,prompt):
        self.prompts.append(prompt); return self
    def to_dict(self):
        return {
            "name":self.name,
            "version":self.version,
            "resources":[r.__dict__ for r in self.resources],
            "tools":[t.__dict__ for t in self.tools],
            "prompts":[p.__dict__ for p in self.prompts],
        }

def mcp_request(method,params=None,request_id=1):
    return {
        "jsonrpc":"2.0",
        "id":request_id,
        "method":method,
        "params":params or {},
    }

def mcp_response(result,request_id=1):
    return {"jsonrpc":"2.0","id":request_id,"result":result}
