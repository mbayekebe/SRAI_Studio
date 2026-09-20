from __future__ import annotations

class AgenticRAGController:
    def __init__(self,retriever,grader=None):
        self.retriever=retriever
        self.grader=grader or (lambda query,hits: len(hits)>0)
    def run(self,query,k=3,max_rounds=3):
        trace=[]; current=query
        for round_no in range(1,max_rounds+1):
            hits=self.retriever(current,k)
            relevant=self.grader(current,hits)
            trace.append({"round":round_no,"query":current,"hits":hits,"relevant":bool(relevant)})
            if relevant: return {"status":"grounded","hits":hits,"trace":trace}
            current=f"{query} detailed evidence sources"
        return {"status":"insufficient","hits":[],"trace":trace}

def retrieval_decision(question):
    lower=str(question).lower()
    return any(term in lower for term in ["current","latest","today","according to the report","document","source"])

def evidence_grade(query,hits,min_score=.1):
    return bool(hits and max(float(h.get("score",0)) for h in hits)>=min_score)

def corrective_query(query,failed_terms):
    return f"{query} {' '.join(failed_terms)}".strip()
