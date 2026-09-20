from __future__ import annotations
import numpy as np

def fuse_embeddings(text_embedding,image_embedding,method="concat"):
    t=np.asarray(text_embedding,float); i=np.asarray(image_embedding,float)
    if method=="concat": return np.concatenate([t,i],axis=-1)
    if method=="mean":
        if t.shape!=i.shape: raise ValueError("Mean fusion requires matching shapes.")
        return (t+i)/2
    raise ValueError("Unsupported fusion method.")

def cross_modal_similarity(text_embeddings,image_embeddings):
    T=np.asarray(text_embeddings,float); I=np.asarray(image_embeddings,float)
    T=T/np.clip(np.linalg.norm(T,axis=1,keepdims=True),1e-12,None)
    I=I/np.clip(np.linalg.norm(I,axis=1,keepdims=True),1e-12,None)
    return T@I.T

def image_grounded_prompt(question,image_description,ocr_text=None):
    prompt=f"Question:\n{question}\n\nImage description:\n{image_description}\n"
    if ocr_text: prompt+=f"\nVisible text:\n{ocr_text}\n"
    return prompt+"\nAnswer only from the visual evidence provided."

def modality_missing_mask(text_present,image_present):
    return {"text_present":bool(text_present),"image_present":bool(image_present),"complete":bool(text_present and image_present)}
