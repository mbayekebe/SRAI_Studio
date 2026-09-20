"""Bayesian decision-analysis utilities."""
from __future__ import annotations
import numpy as np

def posterior(prior,likelihood):
    p=np.asarray(prior,float)
    l=np.asarray(likelihood,float)
    unnormalized=p*l
    return unnormalized/unnormalized.sum()

def expected_loss(loss_matrix,state_probabilities):
    L=np.asarray(loss_matrix,float)
    p=np.asarray(state_probabilities,float)
    return L@p

def bayes_action(loss_matrix,state_probabilities,actions=None):
    losses=expected_loss(loss_matrix,state_probabilities)
    index=int(np.argmin(losses))
    return {
        "action":actions[index] if actions else index,
        "expected_loss":float(losses[index]),
        "all_expected_losses":losses,
    }

def expected_value_perfect_information(payoff_matrix,state_probabilities):
    P=np.asarray(payoff_matrix,float)
    probs=np.asarray(state_probabilities,float)
    current=np.max(P@probs)
    perfect=np.sum(np.max(P,axis=0)*probs)
    return float(perfect-current)

def posterior_predictive(state_values,posterior_probabilities):
    return float(np.sum(np.asarray(state_values,float)*np.asarray(posterior_probabilities,float)))
