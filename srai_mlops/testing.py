import numpy as np
def assert_shape(array,expected_shape):
    actual=tuple(np.asarray(array).shape)
    if actual!=tuple(expected_shape): raise AssertionError(f"Expected {expected_shape}, got {actual}")
    return True
def assert_no_nan(array):
    if np.isnan(np.asarray(array,float)).any(): raise AssertionError("NaN found")
    return True
def prediction_tolerance(actual,expected,tolerance=1e-6):
    return abs(float(actual)-float(expected))<=tolerance
def data_contract_check(frame,required_columns):
    missing=[c for c in required_columns if c not in frame.columns]
    return {"valid":not missing,"missing_columns":missing}
def model_regression_check(current_metric,baseline_metric,max_degradation=.01,higher_is_better=True):
    d=(baseline_metric-current_metric) if higher_is_better else (current_metric-baseline_metric)
    return {"passed":d<=max_degradation,"degradation":float(d)}
