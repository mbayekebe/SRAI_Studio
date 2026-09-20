from dataclasses import dataclass,field
from datetime import datetime,timedelta
@dataclass
class AirflowTask:
    task_id:str
    operator:str="PythonOperator"
    retries:int=1
    retry_delay_minutes:int=5
    upstream:list=field(default_factory=list)
@dataclass
class DAGSpec:
    dag_id:str
    schedule:str
    start_date:str
    catchup:bool=False
    tasks:list=field(default_factory=list)
    def add_task(self,task):
        self.tasks.append(task); return self
    def execution_order(self):
        task_map={t.task_id:t for t in self.tasks}; order=[]; visited=set()
        def visit(task_id):
            if task_id in visited: return
            for dep in task_map[task_id].upstream: visit(dep)
            visited.add(task_id); order.append(task_id)
        for task in self.tasks: visit(task.task_id)
        return order
def backfill_dates(start_date,end_date,frequency_days=1):
    start=datetime.fromisoformat(start_date); end=datetime.fromisoformat(end_date)
    out=[]; current=start
    while current<=end:
        out.append(current.date().isoformat()); current+=timedelta(days=frequency_days)
    return out
def airflow_dag_stub(dag):
    return {"dag_id":dag.dag_id,"schedule":dag.schedule,"start_date":dag.start_date,"catchup":dag.catchup,"tasks":[t.__dict__ for t in dag.tasks]}
