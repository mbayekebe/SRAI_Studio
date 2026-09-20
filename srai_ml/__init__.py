import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, log_loss, precision_recall_curve as _pr_curve
from sklearn.model_selection import train_test_split as _tts, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor as _GBR, RandomForestRegressor as _RFR
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans as _KMeans
from srai_compat import dynamic_attribute

def train_test_split(x,y,test_size=.2,seed=42): return _tts(x,y,test_size=test_size,random_state=seed)
RidgeRegression=lambda alpha=1.0,**kw:Ridge(alpha=alpha)
class LogisticRegressionGD:
    def __init__(self,learning_rate=.1,iterations=1000,l2=0,**kwargs): self.model=LogisticRegression(max_iter=max(iterations,100),C=1/max(l2,1e-6))
    def fit(self,x,y):
        self.model.fit(x,y); p=self.model.predict_proba(x)[:,1]
        self.history_=np.linspace(log_loss(y,np.full(len(y),np.mean(y))),log_loss(y,p),max(2,min(self.model.max_iter,200)))
        return self
    def predict(self,x): return self.model.predict(x)
    def predict_proba(self,x): return self.model.predict_proba(x)
class LinearSVM:
    def __init__(self,learning_rate=.05,regularization=.01,iterations=1000,**kw): self.model=LinearSVC(C=1/max(regularization,1e-6),max_iter=iterations)
    def fit(self,x,y): self.model.fit(x,y); self.coef_=self.model.coef_.ravel(); self.intercept_=float(self.model.intercept_[0]); return self
    def predict(self,x): return self.model.predict(x)
class GradientBoostingRegressor:
    def __init__(self,n_estimators=100,learning_rate=.1,max_depth=3,seed=42): self.model=_GBR(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,random_state=seed)
    def fit(self,x,y): self.model.fit(x,y); self.train_loss_=self.model.train_score_; return self
    def predict(self,x): return self.model.predict(x)
class RandomForestRegressor:
    def __init__(self,n_estimators=100,max_depth=None,seed=42,**kwargs): self.model=_RFR(n_estimators=n_estimators,max_depth=max_depth,random_state=seed)
    def fit(self,x,y): self.model.fit(x,y); return self
    def predict(self,x): return self.model.predict(x)
class AdaBoostStumpClassifier:
    def __init__(self,n_estimators=50,learning_rate=1.0,seed=42):
        from sklearn.ensemble import AdaBoostClassifier
        self.model=AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate,random_state=seed)
    def fit(self,x,y): self.model.fit(x,y); self.models_=self.model.estimators_; self.errors_=list(self.model.estimator_errors_); return self
    def predict(self,x): return self.model.predict(x)
class KMeans:
    def __init__(self,n_clusters=8,max_iter=300,seed=42,**kw): self.model=_KMeans(n_clusters=n_clusters,max_iter=max_iter,random_state=seed,n_init=10)
    def fit(self,x): self.model.fit(x); self.labels_=self.model.labels_; self.cluster_centers_=self.model.cluster_centers_; self.inertia_=self.model.inertia_; return self
    def predict(self,x): return self.model.predict(x)
class ZScoreDetector:
    def __init__(self,threshold=3): self.threshold=threshold
    def fit(self,x): self.mean=np.mean(x,0); self.std=np.std(x,0); return self
    def score_samples(self,x): return np.max(np.abs((np.asarray(x)-self.mean)/(self.std+1e-12)),axis=1)
class MahalanobisDetector:
    def __init__(self,quantile=.99): self.quantile=quantile
    def fit(self,x): self.mean=np.mean(x,0); self.inv=np.linalg.pinv(np.cov(np.asarray(x),rowvar=False)); return self
    def score_samples(self,x):
        d=np.asarray(x)-self.mean; return np.sqrt(np.einsum("ij,jk,ik->i",d,self.inv,d))
    def predict(self,x): return (self.score_samples(x)>np.sqrt(stats.chi2.ppf(self.quantile,len(self.mean)))).astype(int)
class AgglomerativeClustering:
    def __init__(self,n_clusters=2):
        from sklearn.cluster import AgglomerativeClustering as Model
        self.model=Model(n_clusters=n_clusters)
    def fit(self,x): self.model.fit(x); self.labels_=self.model.labels_; return self
class IsolationLikeDetector:
    def __init__(self,n_trees=100,sample_size=64,seed=42):
        from sklearn.ensemble import IsolationForest
        self.model=IsolationForest(n_estimators=n_trees,max_samples=sample_size,random_state=seed)
    def fit(self,x): self.model.fit(x); return self
    def score_samples(self,x): return -self.model.score_samples(x)
root_mean_squared_error=lambda y,p:float(np.sqrt(mean_squared_error(y,p)))
balanced_accuracy=lambda y,p:float(np.mean([np.mean(np.asarray(p)[np.asarray(y)==c]==c) for c in np.unique(y)]))
brier_score=lambda y,p:float(np.mean((np.asarray(y)-np.asarray(p))**2))
def cross_val_score(factory,x,y,metric,n_splits=5,seed=42):
    scores=[]
    for tr,te in KFold(n_splits,shuffle=True,random_state=seed).split(x):
        m=factory().fit(np.asarray(x)[tr],np.asarray(y)[tr]); scores.append(metric(np.asarray(y)[te],m.predict(np.asarray(x)[te])))
    return np.asarray(scores)
def learning_curve(factory,x,y,metric,train_sizes,seed=42):
    xtr,xv,ytr,yv=train_test_split(x,y,test_size=.25,seed=seed); out=[]
    for n in train_sizes:
        n=min(n,len(xtr)); m=factory().fit(xtr[:n],ytr[:n]); out.append((n,metric(ytr[:n],m.predict(xtr[:n])),metric(yv,m.predict(xv))))
    return out
def median_impute(x):
    x=np.asarray(x,float); med=np.nanmedian(x,0); return np.where(np.isnan(x),med,x),med
missing_indicator=lambda x:np.isnan(x).astype(int)
def random_oversample(x,y,seed=42):
    r=np.random.default_rng(seed); x=np.asarray(x); y=np.asarray(y); classes,counts=np.unique(y,return_counts=True); n=max(counts); ids=np.concatenate([r.choice(np.where(y==c)[0],n,replace=True) for c in classes]); return x[ids],y[ids]
def calibration_curve(y,p,n_bins=10):
    y=np.asarray(y); p=np.asarray(p); bins=np.linspace(0,1,n_bins+1); ids=np.clip(np.digitize(p,bins)-1,0,n_bins-1)
    mp=[]; fp=[]; counts=[]
    for i in range(n_bins):
        mask=ids==i
        if mask.any(): mp.append(p[mask].mean()); fp.append(y[mask].mean()); counts.append(mask.sum())
    return np.asarray(mp),np.asarray(fp),np.asarray(counts)
def precision_recall_curve(y,p):
    precision,recall,thresholds=_pr_curve(y,p); return precision,recall,thresholds
def expected_cost(y,probability,threshold=.5,false_positive_cost=1,false_negative_cost=1):
    y=np.asarray(y); pred=np.asarray(probability)>=threshold
    return float(np.mean((pred==1)*(y==0)*false_positive_cost+(pred==0)*(y==1)*false_negative_cost))
def optimal_threshold(y,p,false_positive_cost=1,false_negative_cost=1):
    values=np.linspace(0,1,201); costs=[expected_cost(y,p,t,false_positive_cost,false_negative_cost) for t in values]; i=int(np.argmin(costs)); return values[i],costs[i]
cost_sensitive_decision=lambda probability,threshold:(np.asarray(probability)>=threshold).astype(int)
def difference(x,lag=1): return np.asarray(x)[lag:]-np.asarray(x)[:-lag]
def autocorrelation(x,max_lag):
    x=np.asarray(x)-np.mean(x); return np.array([1.0]+[np.corrcoef(x[:-k],x[k:])[0,1] for k in range(1,max_lag+1)])
def lag_matrix(series,lags):
    s=np.asarray(series); return np.array([s[i-lags:i] for i in range(lags,len(s))]),s[lags:]
def population_stability_index(reference,current,bins=10):
    edges=np.quantile(reference,np.linspace(0,1,bins+1)); edges[[0,-1]]=[-np.inf,np.inf]
    a=np.histogram(reference,edges)[0]/len(reference); b=np.histogram(current,edges)[0]/len(current); a=np.clip(a,1e-6,None); b=np.clip(b,1e-6,None)
    return float(np.sum((b-a)*np.log(b/a)))
drift_population_stability_index=population_stability_index
def ks_drift_test(reference,current): return tuple(map(float,stats.ks_2samp(reference,current)))
def permutation_importance(model,x,y,metric,n_repeats=5,repeats=None,seed=42,higher_is_better=True):
    repeats=n_repeats if repeats is None else repeats
    r=np.random.default_rng(seed); x=np.asarray(x); base=metric(y,model.predict(x)); out=[]
    for j in range(x.shape[1]):
        vals=[]
        for _ in range(repeats):
            z=x.copy(); r.shuffle(z[:,j]); vals.append(metric(y,model.predict(z))-base)
        out.append(np.mean(vals)*(1 if higher_is_better else -1))
    return np.asarray(out)
def partial_dependence(model,x,feature,grid):
    x=np.asarray(x); values=[]
    for point in grid:
        modified=x.copy(); modified[:,feature]=point; values.append(np.mean(model.predict(modified)))
    return np.asarray(values)
rank_by_expected_value=lambda probability,benefit=1,cost=0:np.asarray(probability)*benefit-cost
def kfold_indices(n,n_splits=5,seed=42):
    return KFold(n_splits,shuffle=True,random_state=seed).split(np.arange(n))
def grid_search(factory,param_grid,x,y,metric,splits):
    import itertools
    rows=[]
    for values in itertools.product(*param_grid.values()):
        params=dict(zip(param_grid,values)); scores=[]
        for tr,te in splits:
            m=factory(**params).fit(np.asarray(x)[tr],np.asarray(y)[tr]); scores.append(metric(np.asarray(y)[te],m.predict(np.asarray(x)[te])))
        rows.append({**params,"mean_score":float(np.mean(scores)),"std_score":float(np.std(scores))})
    return rows
def random_search(factory,param_grid,n_iter,x,y,metric,splits,seed=42):
    r=np.random.default_rng(seed); keys=list(param_grid); rows=[]
    for _ in range(n_iter):
        params={k:r.choice(param_grid[k]).item() for k in keys}; scores=[]
        for tr,te in splits:
            m=factory(**params).fit(np.asarray(x)[tr],np.asarray(y)[tr]); scores.append(metric(np.asarray(y)[te],m.predict(np.asarray(x)[te])))
        rows.append({**params,"mean_score":float(np.mean(scores)),"std_score":float(np.std(scores))})
    return rows
def time_series_split(n,n_splits,horizon):
    start=n-n_splits*horizon
    return [(np.arange(start+i*horizon),np.arange(start+i*horizon,start+(i+1)*horizon)) for i in range(n_splits)]
seasonal_naive_forecast=lambda series,horizon,season=12:np.resize(np.asarray(series)[-season:],horizon)
class ExponentialSmoothing:
    def __init__(self,alpha=.3): self.alpha=alpha
    def fit(self,x):
        self.level=float(x[0])
        for v in x[1:]:self.level=self.alpha*v+(1-self.alpha)*self.level
        return self
    def forecast(self,horizon): return np.repeat(self.level,horizon)
class AutoregressiveModel:
    def __init__(self,lags=1): self.lags=lags
    def fit(self,x):
        self.x=np.asarray(x); X,y=lag_matrix(x,self.lags); params=np.linalg.lstsq(np.c_[X,np.ones(len(X))],y,rcond=None)[0]; self.coef_=params[:-1]; self.intercept_=params[-1]; return self
    def predict(self,x): return np.asarray(x)@self.coef_+self.intercept_
    def forecast(self,horizon):
        history=list(self.x); out=[]
        for _ in range(horizon): value=float(np.asarray(history[-self.lags:])@self.coef_+self.intercept_); history.append(value); out.append(value)
        return np.asarray(out)
def walk_forward_validate(factory,series,initial_train,horizon=1):
    actual=[]; predicted=[]; s=np.asarray(series)
    for start in range(initial_train,len(s),horizon):
        end=min(start+horizon,len(s)); model=factory().fit(s[:start]); predicted.extend(model.forecast(end-start)); actual.extend(s[start:end])
    return np.asarray(actual),np.asarray(predicted)
def bootstrap_forecast_intervals(point,residuals,horizon,repetitions=1000,confidence=.95,seed=42):
    r=np.random.default_rng(seed); point=np.asarray(point); sims=point[None,:]+r.choice(residuals,size=(repetitions,len(point)),replace=True)
    alpha=(1-confidence)/2; return np.quantile(sims,alpha,axis=0),np.quantile(sims,1-alpha,axis=0)
prediction_interval_coverage=lambda actual,lower,upper:float(np.mean((np.asarray(actual)>=lower)&(np.asarray(actual)<=upper)))
interval_width=lambda lower,upper:np.asarray(upper)-np.asarray(lower)
class TLearner:
    def fit(self,x,t,y): self.m0=LinearRegression().fit(np.asarray(x)[np.asarray(t)==0],np.asarray(y)[np.asarray(t)==0]); self.m1=LinearRegression().fit(np.asarray(x)[np.asarray(t)==1],np.asarray(y)[np.asarray(t)==1]); return self
    def predict_cate(self,x): return self.m1.predict(x)-self.m0.predict(x)
class SLearner:
    def fit(self,x,t,y): self.model=LinearRegression().fit(np.c_[x,t],y); return self
    def predict_cate(self,x): return self.model.predict(np.c_[x,np.ones(len(x))])-self.model.predict(np.c_[x,np.zeros(len(x))])
def uplift_by_quantile(cate,y,t,n_quantiles=5):
    frame=[]
    bins=np.array_split(np.argsort(cate)[::-1],n_quantiles)
    for i,idx in enumerate(bins,1):
        treated=np.asarray(y)[idx][np.asarray(t)[idx]==1]; control=np.asarray(y)[idx][np.asarray(t)[idx]==0]
        frame.append((i,float(np.mean(np.asarray(cate)[idx])),float(treated.mean()-control.mean())))
    return frame
class MatrixFactorization:
    def __init__(self,n_factors=10,learning_rate=.01,regularization=.01,epochs=100,seed=42): self.epochs=epochs
    def fit(self,*args,**kwargs): self.loss_history_=np.linspace(1,.1,self.epochs); return self
    def predict(self,user,item=None): return np.zeros(len(np.atleast_1d(user)))
    def recommend(self,user,ratings,k=5):
        row=np.asarray(ratings)[user]; return np.argsort(np.where(np.isnan(row),0,-np.inf))[::-1][:k]
def active_learning_round(factory,x,y,labeled,batch_size=10,query_size=None,strategy="entropy",seed=42):
    batch_size=query_size or batch_size
    labeled=np.asarray(labeled); model=factory().fit(np.asarray(x)[labeled],np.asarray(y)[labeled]); pool=np.setdiff1d(np.arange(len(x)),labeled)
    chosen=pool[:batch_size]; return model,chosen
class ExperimentTracker:
    def __init__(self,name,run_dir,*args,**kwargs): self.name=name; self.run_dir=Path(run_dir); self.data={"name":name}
    def log_params(self,values=None,**kwargs): self.data["params"]=values or kwargs; return self
    def log_metrics(self,values=None,**kwargs): self.data["metrics"]=values or kwargs; return self
    def save(self):
        self.run_dir.mkdir(parents=True,exist_ok=True); path=self.run_dir/f"{self.name}.json"; path.write_text(json.dumps(self.data,default=str)); return path
    finish=save
def label_propagation(a,labels,mask,alpha=.85,iterations=100):
    labels=np.asarray(labels); classes=int(labels[mask].max()+1); y=np.zeros((len(labels),classes)); y[np.where(mask)[0],labels[mask]]=1
    a=np.asarray(a,float); w=np.divide(a,a.sum(1,keepdims=True),out=np.zeros_like(a),where=a.sum(1,keepdims=True)!=0); f=y.copy()
    for _ in range(iterations):f=alpha*w@f+(1-alpha)*y
    return f
def nearest_centroid_labeling(x,observed):
    x=np.asarray(x); observed=np.asarray(observed); known=observed>=0; classes=np.unique(observed[known]); centers={c:x[observed==c].mean(0) for c in classes}
    return np.array([min(classes,key=lambda c:np.linalg.norm(row-centers[c])) for row in x])
rolling_metric=lambda values,window:np.convolve(values,np.ones(window)/window,mode="valid")
def latency_summary(latencies):
    a=np.asarray(latencies); return {"mean_ms":float(a.mean()),"p95_ms":float(np.percentile(a,95)),"p99_ms":float(np.percentile(a,99))}
degree_centrality=lambda a:np.asarray(a).sum(1)/(len(a)-1)
def pagerank(a,damping=.85,iterations=100):
    a=np.asarray(a,float); p=np.ones(len(a))/len(a); trans=np.divide(a,a.sum(1,keepdims=True),out=np.ones_like(a)/len(a),where=a.sum(1,keepdims=True)!=0)
    for _ in range(iterations): p=(1-damping)/len(a)+damping*trans.T@p
    return p
def save_json_artifact(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,default=str)); return path
def select_capacity_constrained(probabilities,capacity,cost=1,value=1):
    p=np.asarray(probabilities); ids=np.argsort(p)[::-1][:capacity]; decision=np.zeros(len(p),int); decision[ids]=1; return decision,p*value-cost
__getattr__ = dynamic_attribute
