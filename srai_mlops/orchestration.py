from dataclasses import dataclass,field
@dataclass
class Task:
    name:str
    function:object
    dependencies:list=field(default_factory=list)
class Workflow:
    def __init__(self,tasks): self.tasks={t.name:t for t in tasks}
    def execution_order(self):
        order=[]; visited=set()
        def visit(name):
            if name in visited: return
            for dep in self.tasks[name].dependencies: visit(dep)
            visited.add(name); order.append(name)
        for name in self.tasks: visit(name)
        return order
    def run(self,context=None):
        context={} if context is None else dict(context); results={}
        for name in self.execution_order():
            task=self.tasks[name]
            results[name]=task.function(context,{d:results[d] for d in task.dependencies})
        return results
def retry(function,attempts=3):
    error=None
    for _ in range(attempts):
        try: return function()
        except Exception as exc: error=exc
    raise error
