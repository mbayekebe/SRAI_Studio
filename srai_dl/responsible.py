"""Responsible deep-learning utilities."""
from __future__ import annotations
import numpy as np

def demographic_parity_gap(predictions,groups):
    p=np.asarray(predictions,int); g=np.asarray(groups)
    rates=[np.mean(p[g==group]==1) for group in np.unique(g)]
    return float(max(rates)-min(rates)) if rates else 0.0

def equal_opportunity_gap(y_true,predictions,groups):
    y=np.asarray(y_true,int); p=np.asarray(predictions,int); g=np.asarray(groups)
    recalls=[]
    for group in np.unique(g):
        mask=(g==group)&(y==1)
        recalls.append(np.mean(p[mask]==1) if np.any(mask) else 0.0)
    return float(max(recalls)-min(recalls)) if recalls else 0.0

def adversarial_perturbation(gradient,epsilon=.01):
    return epsilon*np.sign(np.asarray(gradient,float))

def confidence_abstention(probabilities,threshold=.6):
    p=np.asarray(probabilities,float)
    confidence=np.max(p,axis=1)
    prediction=np.argmax(p,axis=1)
    prediction[confidence<threshold]=-1
    return prediction,confidence

def privacy_risk_score(train_losses,test_losses):
    train=np.asarray(train_losses,float); test=np.asarray(test_losses,float)
    return float(max(0.0,np.mean(test)-np.mean(train)))

def responsible_ai_checklist():
    return [
        "document_training_data",
        "evaluate_subgroups",
        "test_robustness",
        "calibrate_confidence",
        "define_abstention_policy",
        "assess_privacy",
        "maintain_human_oversight",
        "monitor_post_deployment_harm",
    ]
