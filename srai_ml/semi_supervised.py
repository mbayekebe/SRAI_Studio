"""Semi-supervised learning utilities."""
from __future__ import annotations
import numpy as np

class SelfTrainingClassifier:
    def __init__(self,base_model_factory,threshold=0.9,max_iter=10):
        self.base_model_factory=base_model_factory
        self.threshold=threshold
        self.max_iter=max_iter
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        labeled=y>=0
        current=y.copy()
        self.history_=[]
        for _ in range(self.max_iter):
            model=self.base_model_factory().fit(X[labeled],current[labeled])
            if not hasattr(model,"predict_proba"):
                break
            unlabeled=np.where(~labeled)[0]
            if unlabeled.size==0:
                break
            proba=model.predict_proba(X[unlabeled])
            confidence=np.max(proba,axis=1)
            pseudo=np.argmax(proba,axis=1)
            selected=unlabeled[confidence>=self.threshold]
            if selected.size==0:
                break
            current[selected]=pseudo[confidence>=self.threshold]
            labeled[selected]=True
            self.history_.append(int(selected.size))
        self.model_=self.base_model_factory().fit(X[labeled],current[labeled])
        self.transduced_labels_=current
        return self
    def predict(self,X):
        return self.model_.predict(X)

def nearest_centroid_labeling(X,y):
    X=np.asarray(X,float); y=np.asarray(y,int)
    labeled=y>=0
    classes=np.unique(y[labeled])
    centroids=np.array([X[(y==c)&labeled].mean(axis=0) for c in classes])
    unlabeled=np.where(~labeled)[0]
    result=y.copy()
    for i in unlabeled:
        result[i]=classes[np.argmin(np.linalg.norm(centroids-X[i],axis=1))]
    return result

def consistency_loss(probabilities_a,probabilities_b):
    a=np.asarray(probabilities_a,float); b=np.asarray(probabilities_b,float)
    return float(np.mean((a-b)**2))
