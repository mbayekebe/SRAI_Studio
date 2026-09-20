from dataclasses import dataclass,asdict
import numpy as np
@dataclass
class InferencePayload:
    request_id:str
    features:list
@dataclass
class InferenceResult:
    request_id:str
    prediction:float
    model_version:str
def validate_payload(payload,expected_features):
    errors=[]
    if not payload.request_id: errors.append("missing_request_id")
    if len(payload.features)!=expected_features: errors.append("invalid_feature_count")
    if not all(isinstance(x,(int,float)) for x in payload.features): errors.append("non_numeric_feature")
    return {"valid":not errors,"errors":errors}
def build_inference_handler(predict_function,model_version,expected_features):
    def handler(payload):
        v=validate_payload(payload,expected_features)
        if not v["valid"]: return {"status_code":422,"detail":v["errors"]}
        pred=float(np.asarray(predict_function(np.asarray(payload.features,float))).ravel()[0])
        return {"status_code":200,"body":asdict(InferenceResult(payload.request_id,pred,model_version))}
    return handler
def openapi_stub(title,version):
    return {"openapi":"3.1.0","info":{"title":title,"version":version},"paths":{"/health":{"get":{}},"/predict":{"post":{}}}}
