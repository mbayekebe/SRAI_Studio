"""Tokenization utilities for Generative AI demonstrations."""
from __future__ import annotations
import re
import numpy as np

class WordTokenizer:
    def __init__(self,special_tokens=("<PAD>","<UNK>","<BOS>","<EOS>")):
        self.special_tokens=list(special_tokens)
    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+|[^\w\s]",str(text).lower())
    def fit(self,texts):
        vocab=sorted(set(token for text in texts for token in self.tokenize(text)))
        self.itos_=self.special_tokens+vocab
        self.stoi_={t:i for i,t in enumerate(self.itos_)}
        return self
    def encode(self,text,add_special_tokens=True):
        ids=[self.stoi_.get(t,self.stoi_["<UNK>"]) for t in self.tokenize(text)]
        if add_special_tokens:
            ids=[self.stoi_["<BOS>"]]+ids+[self.stoi_["<EOS>"]]
        return np.asarray(ids,dtype=int)
    def decode(self,ids,skip_special_tokens=True):
        tokens=[self.itos_[int(i)] for i in ids]
        if skip_special_tokens:
            tokens=[t for t in tokens if t not in self.special_tokens]
        text=" ".join(tokens)
        return re.sub(r"\s+([,.;:!?])",r"\1",text)

def whitespace_tokens(text):
    return str(text).split()

def token_statistics(texts,tokenizer):
    counts=[len(tokenizer.encode(t,add_special_tokens=False)) for t in texts]
    return {
        "documents":len(texts),
        "total_tokens":int(sum(counts)),
        "mean_tokens":float(np.mean(counts)),
        "max_tokens":int(max(counts) if counts else 0),
    }
