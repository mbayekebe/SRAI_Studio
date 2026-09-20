from __future__ import annotations
import numpy as np
class BigramLanguageModel:
 def __init__(self,vocab_size,alpha=1.): self.vocab_size=vocab_size; self.alpha=alpha
 def fit(self,sequences):
  c=np.full((self.vocab_size,self.vocab_size),self.alpha)
  for seq in sequences:
   for a,b in zip(seq[:-1],seq[1:]): c[int(a),int(b)]+=1
  self.prob_=c/c.sum(1,keepdims=True); return self
 def generate(self,start_token,length,seed=42):
  r=np.random.default_rng(seed); out=[int(start_token)]
  for _ in range(length-1): out.append(int(r.choice(self.vocab_size,p=self.prob_[out[-1]])))
  return out
 def sequence_log_probability(self,seq): return float(sum(np.log(self.prob_[int(a),int(b)]) for a,b in zip(seq[:-1],seq[1:])))
def perplexity(log_probability,token_count): return float(np.exp(-log_probability/token_count))
def temperature_scale(logits,temperature=1.):
 x=np.asarray(logits,float)/temperature; x-=x.max(); e=np.exp(x); return e/e.sum()
def top_k_sampling(probabilities,k,seed=42):
 p=np.asarray(probabilities,float); idx=np.argsort(p)[::-1][:min(k,len(p))]; q=p[idx]/p[idx].sum(); return int(np.random.default_rng(seed).choice(idx,p=q))
