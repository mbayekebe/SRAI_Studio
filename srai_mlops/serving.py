"""Model-serving helpers and API-style contracts."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np

@dataclass
class PredictionRequest:
    request_id:str
    features:list

@dataclass
class PredictionResponse:
    request_id:str
    prediction:float
    model_version:str
    latency_ms:float

class ModelService:
    def __init__(self,predict_function,model_version="1.0.0"):
        self.predict_function=predict_function
        self.model_version=model_version
        self.requests=0
        self.errors=0
        self.latencies=[]
    def predict(self,request):
        start=time.perf_counter()
        try:
            value=self.predict_function(np.asarray(request.features,float))
            prediction=float(np.asarray(value).ravel()[0])
            self.requests+=1
            latency=(time.perf_counter()-start)*1000
            self.latencies.append(latency)
            return PredictionResponse(
                request_id=request.request_id,
                prediction=prediction,
                model_version=self.model_version,
                latency_ms=latency,
            )
        except Exception:
            self.errors+=1
            raise
    def health(self):
        return {
            "status":"ok",
            "model_version":self.model_version,
            "requests":self.requests,
            "errors":self.errors,
        }
    def metrics(self):
        return {
            "requests":self.requests,
            "errors":self.errors,
            "error_rate":self.errors/max(self.requests+self.errors,1),
            "mean_latency_ms":float(np.mean(self.latencies)) if self.latencies else 0.0,
            "p95_latency_ms":float(np.quantile(self.latencies,.95)) if self.latencies else 0.0,
        }

def response_to_dict(response):
    return asdict(response)
