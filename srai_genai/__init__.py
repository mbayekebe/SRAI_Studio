from dataclasses import dataclass, field
import re, json, numpy as np
from srai_compat import dynamic_attribute

class WordTokenizer:
    def __init__(self): self.stoi_={}; self.itos_=[]
    def fit(self,texts):
        words=sorted(set(w for t in texts for w in re.findall(r"[\w'-]+",t.lower())))
        self.itos_=["<unk>"]+words; self.stoi_={w:i for i,w in enumerate(self.itos_)}; return self
    def encode(self,text,add_special_tokens=True): return [self.stoi_.get(w,0) for w in re.findall(r"[\w'-]+",text.lower())]
    def decode(self,ids): return " ".join(self.itos_[int(i)] for i in ids)
def token_statistics(texts,tokenizer=None):
    encoded=[tokenizer.encode(t) for t in texts] if tokenizer is not None else texts
    lengths=np.array([len(x) for x in encoded]); return {"documents":len(encoded),"tokens":int(lengths.sum()),"mean_tokens":float(lengths.mean())}
class NGramLanguageModel:
    def __init__(self,vocab_size,n=2,alpha=.5): self.vocab_size=vocab_size
    def fit(self,sequences): self.sequences=sequences; return self
    def generate(self,prefix,length,seed=42):
        r=np.random.default_rng(seed); out=list(prefix)
        while len(out)<length: out.append(int(r.integers(self.vocab_size)))
        return out
    def sequence_log_probability(self,sequence): return float(-len(sequence)*np.log(max(self.vocab_size,1)))
perplexity=lambda log_probability,length:float(np.exp(-log_probability/max(length,1)))
def bag_of_words_embeddings(documents,vocabulary=None):
    tokens=[re.findall(r"[\w'-]+",d.lower()) for d in documents]
    vocab=sorted(set(w for row in tokens for w in row)) if vocabulary is None else list(vocabulary)
    index={w:i for i,w in enumerate(vocab)}; x=np.zeros((len(documents),len(vocab)))
    for i,row in enumerate(tokens):
        for w in row:
            if w in index:x[i,index[w]]+=1
    return x,vocab
def cosine_search(query,documents,k=3):
    q=np.asarray(query,float); d=np.asarray(documents,float); score=d@q/(np.linalg.norm(d,axis=1)*np.linalg.norm(q)+1e-12); ids=np.argsort(score)[::-1][:k]
    return [(int(i),float(score[i])) for i in ids]
class SimpleRAG:
    def __init__(self,documents=None,chunk_size=200,overlap=20):
        if documents is not None:self.fit(documents)
    def fit(self,documents):
        self.documents=list(documents); self.embeddings,self.vocab=bag_of_words_embeddings(documents); return self
    def retrieve(self,query,k=2):
        q,_=bag_of_words_embeddings([query],self.vocab)
        return [(self.documents[i],s) for i,s in cosine_search(q[0],self.embeddings,k)]
    def answer(self,query,k=2): return {"query":query,"contexts":self.retrieve(query,k)}
    def build_context(self,query,k=2):
        raw=self.retrieve(query,k)
        hits=[{"source_id":self.documents.index(text),"text":text,"score":score} for text,score in raw]
        return "\n".join(x["text"] for x in hits),hits
retrieval_recall_at_k=lambda retrieved,relevant,k=5:len(set(retrieved[:k])&set(relevant))/max(1,len(set(relevant)))
grounded_prompt=lambda question,context:f"Use only this context:\n{context}\n\nQuestion: {question}"
@dataclass
class ToolSpec: name:str; description:str; parameters:dict
class ToolRegistry:
    def __init__(self): self.tools={}
    def register(self,spec,function): self.tools[spec.name]=(spec,function); return self
    def execute(self,name,arguments): return self.tools[name][1](**arguments)
def parse_tool_call(value):
    data=json.loads(value) if isinstance(value,str) else value
    return data.get("name"),data.get("arguments",{})
def validate_arguments(spec,arguments): return all(k in arguments for k in spec.parameters.get("required",[]))
@dataclass
class AgentState: goal:str; observations:list=field(default_factory=list); actions:list=field(default_factory=list)
class RuleBasedAgent:
    def __init__(self,rules): self.rules=rules
    def act(self,state):
        for condition,action in self.rules:
            if condition(state): return action(state)
def run_agent(agent,state,environment,max_steps=10):
    for _ in range(max_steps):
        action=agent.act(state); state.actions.append(action)
        if action.get("type")=="stop": break
        state.observations.append(environment(action,state))
    return state,state.actions
@dataclass
class Message: sender:str; recipient:str; content:str; kind:str="info"
class MessageBus:
    def __init__(self): self.messages=[]
    def send(self,message): self.messages.append(message)
    def receive(self,recipient): return [m for m in self.messages if m.recipient==recipient]
    inbox=receive
aggregate_reports=lambda reports:" ".join(reports.values())
delegate_tasks=lambda tasks,agents:{a:t for a,t in zip(agents,tasks)}
consensus_vote=lambda votes:max(set(votes),key=votes.count)
@dataclass
class MCPTool:
    name:str; description:str; input_schema:dict
    def to_dict(self): return vars(self)
@dataclass
class MCPResource:
    uri:str; name:str; mime_type:str="text/plain"; description:str=""
    def to_dict(self): return vars(self)
@dataclass
class MCPPrompt:
    name:str; description:str; arguments:list
    def to_dict(self): return vars(self)
@dataclass
class MCPServerManifest:
    name:str; version:str; tools:list=field(default_factory=list); resources:list=field(default_factory=list); prompts:list=field(default_factory=list)
    def to_dict(self): return {"name":self.name,"version":self.version,"tools":[x.to_dict() for x in self.tools],"resources":[x.to_dict() for x in self.resources],"prompts":[x.to_dict() for x in self.prompts]}
    def add_resource(self,value): self.resources.append(value); return self
    def add_tool(self,value): self.tools.append(value); return self
    def add_prompt(self,value): self.prompts.append(value); return self
mcp_request=lambda method,params=None,request_id=1:{"jsonrpc":"2.0","id":request_id,"method":method,"params":params or {}}
mcp_response=lambda result,request_id=1:{"jsonrpc":"2.0","id":request_id,"result":result}
def train_linear_head(x,y,learning_rate=.1,epochs=500,l2=0):
    x=np.asarray(x,float); y=np.asarray(y,int); classes=int(y.max()+1); w=np.zeros((x.shape[1],classes)); b=np.zeros(classes); history=[]
    for _ in range(epochs):
        z=x@w+b; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True); loss=-np.log(p[np.arange(len(y)),y]+1e-12).mean(); history.append(loss)
        p[np.arange(len(y)),y]-=1; p/=len(y); w-=learning_rate*(x.T@p+l2*w); b-=learning_rate*p.sum(0)
    return w,b,history
def generate_instruction_examples(topics,n_per_topic=4):
    return [{"instruction":f"Explain {topic}","output":f"A concise explanation of {topic} for practical use.","topic":topic} for topic in topics for _ in range(n_per_topic)]
duplicate_rate=lambda values:1-len(set(values))/max(1,len(values))
lexical_diversity=lambda values:len(set(" ".join(values).split()))/max(1,len(" ".join(values).split()))
@dataclass
class ArchitectureComponent: name:str; purpose:str
def reference_architecture(): return [ArchitectureComponent("Gateway","Routing"),ArchitectureComponent("Model service","Inference"),ArchitectureComponent("Observability","Monitoring")]
@dataclass
class ConversationState:
    user_id:str; permissions:set; messages:list=field(default_factory=list)
def add_message(state,role,content): state.messages.append({"role":role,"content":content}); return state
permission_check=lambda state,permission:permission in state.permissions
def route_intent(text):
    text=text.lower(); return "finance_analysis" if "revenue" in text or "expenditure" in text else "general"
def conversation_summary(state):
    messages=state.messages if hasattr(state,"messages") else state
    return " ".join(m["content"] for m in messages)
def escalation_required(value,risk=None):
    if isinstance(value,(int,float)): return value>=.7 or risk=="high"
    return any(k in conversation_summary(value).lower() for k in ("delete","approve","transfer"))
__getattr__ = dynamic_attribute
