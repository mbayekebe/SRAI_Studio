"""Small autoregressive language-model utilities."""
from __future__ import annotations
import numpy as np

class NGramLanguageModel:
    def __init__(self,vocab_size,n=2,alpha=1.0):
        if n<1: raise ValueError("n must be >=1")
        self.vocab_size=vocab_size; self.n=n; self.alpha=alpha
    def fit(self,sequences):
        self.counts_={}
        self.context_totals_={}
        for seq in sequences:
            seq=list(map(int,seq))
            padded=[0]*(self.n-1)+seq
            for i in range(self.n-1,len(padded)):
                context=tuple(padded[i-self.n+1:i]) if self.n>1 else ()
                token=padded[i]
                self.counts_[(context,token)]=self.counts_.get((context,token),0)+1
                self.context_totals_[context]=self.context_totals_.get(context,0)+1
        return self
    def next_token_probabilities(self,context):
        context=tuple(list(map(int,context))[-(self.n-1):]) if self.n>1 else ()
        total=self.context_totals_.get(context,0)+self.alpha*self.vocab_size
        return np.array([
            (self.counts_.get((context,t),0)+self.alpha)/total
            for t in range(self.vocab_size)
        ])
    def generate(self,start_context,length,seed=42):
        rng=np.random.default_rng(seed)
        tokens=list(map(int,start_context))
        for _ in range(length):
            p=self.next_token_probabilities(tokens)
            tokens.append(int(rng.choice(self.vocab_size,p=p)))
        return tokens
    def sequence_log_probability(self,sequence):
        seq=list(map(int,sequence))
        padded=[0]*(self.n-1)+seq
        value=0.0
        for i in range(self.n-1,len(padded)):
            context=tuple(padded[i-self.n+1:i]) if self.n>1 else ()
            token=padded[i]
            p=self.next_token_probabilities(context)[token]
            value+=np.log(max(p,1e-12))
        return float(value)

def perplexity(log_probability,token_count):
    return float(np.exp(-log_probability/max(token_count,1)))

def temperature_sampling(probabilities,temperature=1.0,seed=42):
    if temperature<=0: raise ValueError("temperature must be positive.")
    p=np.asarray(probabilities,float)
    logits=np.log(np.clip(p,1e-12,1))/temperature
    logits-=np.max(logits)
    q=np.exp(logits); q/=q.sum()
    return int(np.random.default_rng(seed).choice(len(q),p=q))
