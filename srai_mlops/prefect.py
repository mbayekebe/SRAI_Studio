from dataclasses import dataclass,field
@dataclass
class PrefectTask:
    name:str
    function:object
    retries:int=0
    cache_key:str|None=None
    dependencies:list=field(default_factory=list)
class PrefectFlow:
    def __init__(self,name,tasks):
        self.name=name; self.tasks={t.name:t for t in tasks}
    def execution_order(self):
        order=[]; visited=set()
        def visit(name):
            if name in visited: return
            for dep in self.tasks[name].dependencies: visit(dep)
            visited.add(name); order.append(name)
        for name in self.tasks: visit(name)
        return order
    def run(self,parameters=None):
        parameters=parameters or {}; results={}
        for name in self.execution_order():
            task=self.tasks[name]; deps={d:results[d] for d in task.dependencies}
            error=None
            for _ in range(task.retries+1):
                try:
                    results[name]=task.function(parameters,deps); error=None; break
                except Exception as exc: error=exc
            if error is not None: raise error
        return results
def deployment_spec(flow_name,schedule,work_pool="default"):
    return {"flow_name":flow_name,"schedule":schedule,"work_pool":work_pool}
