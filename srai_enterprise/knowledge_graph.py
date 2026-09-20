"""Enterprise knowledge-graph helpers."""
from __future__ import annotations
from collections import defaultdict, deque

class KnowledgeGraph:
    def __init__(self):
        self.nodes={}
        self.edges=[]
        self.adjacency=defaultdict(list)

    def add_node(self,node_id,node_type,properties=None):
        self.nodes[node_id]={"type":node_type,"properties":properties or {}}
        return self

    def add_edge(self,source,relation,target,properties=None):
        edge={"source":source,"relation":relation,"target":target,"properties":properties or {}}
        self.edges.append(edge)
        self.adjacency[source].append((relation,target))
        return self

    def neighbors(self,node_id,relation=None):
        rows=self.adjacency.get(node_id,[])
        return [target for rel,target in rows if relation is None or rel==relation]

    def shortest_path(self,start,target):
        queue=deque([(start,[start])]); visited={start}
        while queue:
            node,path=queue.popleft()
            if node==target:
                return path
            for _,neighbor in self.adjacency.get(node,[]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor,path+[neighbor]))
        return None

def entity_resolution_score(name_a,name_b):
    a=set(str(name_a).lower().split())
    b=set(str(name_b).lower().split())
    return len(a&b)/max(len(a|b),1)

def graph_summary(graph):
    return {
        "nodes":len(graph.nodes),
        "edges":len(graph.edges),
        "node_types":sorted(set(v["type"] for v in graph.nodes.values())),
    }
