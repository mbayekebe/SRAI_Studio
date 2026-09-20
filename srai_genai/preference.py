"""Preference optimization and RLHF concept utilities."""
from __future__ import annotations
import numpy as np

def pairwise_preference_loss(chosen_score,rejected_score,beta=1.0):
    margin=beta*(np.asarray(chosen_score,float)-np.asarray(rejected_score,float))
    return float(np.mean(np.log1p(np.exp(-margin))))

def reward_model_accuracy(chosen_scores,rejected_scores):
    c=np.asarray(chosen_scores,float)
    r=np.asarray(rejected_scores,float)
    return float(np.mean(c>r))

def dpo_loss(policy_chosen,policy_rejected,reference_chosen,reference_rejected,beta=.1):
    policy_gap=np.asarray(policy_chosen,float)-np.asarray(policy_rejected,float)
    reference_gap=np.asarray(reference_chosen,float)-np.asarray(reference_rejected,float)
    logits=beta*(policy_gap-reference_gap)
    return float(np.mean(np.log1p(np.exp(-logits))))

def kl_penalty(policy_log_probs,reference_log_probs):
    p=np.asarray(policy_log_probs,float)
    r=np.asarray(reference_log_probs,float)
    return float(np.mean(p-r))

def rank_responses(scores):
    return list(np.argsort(np.asarray(scores,float))[::-1])
