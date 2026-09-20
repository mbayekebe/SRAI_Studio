"""Document ingestion and chunking utilities."""
from __future__ import annotations
import re
from dataclasses import dataclass
from .retrieval import chunk_text

@dataclass
class Document:
    text:str
    metadata:dict

def clean_text(text):
    text=str(text).replace("\x00"," ")
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def split_sections(text):
    lines=[line.strip() for line in str(text).splitlines()]
    sections=[]; current=[]
    for line in lines:
        if line and (line.endswith(":") or line.isupper()):
            if current: sections.append(" ".join(current)); current=[]
            current.append(line)
        elif line:
            current.append(line)
    if current: sections.append(" ".join(current))
    return sections

def ingest_texts(texts,source_names=None):
    source_names=source_names or [f"source_{i}" for i in range(len(texts))]
    return [Document(clean_text(text),{"source":name}) for text,name in zip(texts,source_names)]

def chunk_documents(documents,chunk_size=80,overlap=15):
    rows=[]
    for doc_id,doc in enumerate(documents):
        for chunk_id,chunk in enumerate(chunk_text(doc.text,chunk_size,overlap)):
            metadata=dict(doc.metadata)
            metadata.update({"document_id":doc_id,"chunk_id":chunk_id})
            rows.append(Document(chunk,metadata))
    return rows

def chunk_statistics(chunks):
    lengths=[len(c.text.split()) for c in chunks]
    return {
        "chunks":len(chunks),
        "mean_words":sum(lengths)/len(lengths) if lengths else 0,
        "max_words":max(lengths) if lengths else 0,
    }
