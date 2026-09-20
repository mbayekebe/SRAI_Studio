"""Retrieval-augmented generation helpers."""
from __future__ import annotations
import numpy as np
from .retrieval import bag_of_words_embeddings,cosine_search,chunk_text

class SimpleRAG:
    def __init__(self,chunk_size=60,overlap=10):
        self.chunk_size=chunk_size
        self.overlap=overlap
    def fit(self,documents):
        chunks=[]
        sources=[]
        for doc_id,doc in enumerate(documents):
            for chunk in chunk_text(doc,self.chunk_size,self.overlap):
                chunks.append(chunk)
                sources.append(doc_id)
        self.chunks_=chunks
        self.sources_=sources
        self.embeddings_,self.vocabulary_=bag_of_words_embeddings(chunks)
        return self
    def retrieve(self,query,k=3):
        q,_=bag_of_words_embeddings([query],self.vocabulary_)
        results=cosine_search(q[0],self.embeddings_,k)
        return [{
            "chunk_id":i,
            "source_id":self.sources_[i],
            "text":self.chunks_[i],
            "score":score,
        } for i,score in results]
    def build_context(self,query,k=3):
        hits=self.retrieve(query,k)
        return "\n\n".join(
            f"[Source {h['source_id']} | Chunk {h['chunk_id']}] {h['text']}"
            for h in hits
        ),hits

def grounded_prompt(question,context):
    return f"""Answer the question using only the provided context.
If the answer is not supported by the context, say that the context is insufficient.

Question:
{question}

Context:
{context}

Answer with source references."""

def retrieval_recall_at_k(retrieved_ids,relevant_ids,k):
    retrieved=set(retrieved_ids[:k])
    relevant=set(relevant_ids)
    return float(len(retrieved & relevant)/len(relevant)) if relevant else 0.0
