from __future__ import annotations
import re
from .retrieval import bag_of_words_embeddings, cosine_search

def query_expansion(query,synonyms=None):
    synonyms=synonyms or {}
    tokens=re.findall(r"\b\w+\b",str(query).lower())
    expanded=list(tokens)
    for token in tokens: expanded.extend(synonyms.get(token,[]))
    return " ".join(expanded)

def hypothetical_document(query):
    return f"A relevant document answering the question: {query}"

def hybrid_scores(query,documents,semantic_scores,lexical_weight=.4,semantic_weight=.6):
    q=set(re.findall(r"\b\w+\b",str(query).lower()))
    rows=[]
    for i,doc in enumerate(documents):
        d=set(re.findall(r"\b\w+\b",str(doc).lower()))
        lexical=len(q&d)/max(len(q),1)
        score=lexical_weight*lexical+semantic_weight*float(semantic_scores[i])
        rows.append({"index":i,"lexical":lexical,"semantic":float(semantic_scores[i]),"score":score})
    return sorted(rows,key=lambda x:x["score"],reverse=True)

def contextual_compression(query,document,max_sentences=2):
    sentences=re.split(r"(?<=[.!?])\s+",str(document).strip())
    q=set(re.findall(r"\b\w+\b",str(query).lower()))
    scored=[]
    for sentence in sentences:
        terms=set(re.findall(r"\b\w+\b",sentence.lower()))
        scored.append((len(q&terms),sentence))
    scored.sort(key=lambda x:x[0],reverse=True)
    return " ".join(sentence for _,sentence in scored[:max_sentences] if sentence)

def multi_query_retrieve(queries,documents,k=3):
    X,vocab=bag_of_words_embeddings(documents)
    scores={}
    for query in queries:
        q,_=bag_of_words_embeddings([query],vocab)
        for idx,score in cosine_search(q[0],X,k):
            scores[idx]=max(scores.get(idx,-1),score)
    return sorted(scores.items(),key=lambda x:x[1],reverse=True)
