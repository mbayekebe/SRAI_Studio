import numpy as np
from srai_compat import dynamic_attribute

relu = lambda x: np.maximum(np.asarray(x), 0)
sigmoid = lambda x: 1 / (1 + np.exp(-np.asarray(x)))
tanh = np.tanh
softmax = lambda x, axis=-1: np.exp(np.asarray(x)-np.max(x,axis=axis,keepdims=True))/np.sum(np.exp(np.asarray(x)-np.max(x,axis=axis,keepdims=True)),axis=axis,keepdims=True)

class Dense:
    def __init__(self,input_dim,output_dim,seed=42,scale=.1):
        self.W=np.random.default_rng(seed).normal(scale=scale,size=(input_dim,output_dim)); self.b=np.zeros(output_dim)
    def forward(self,x): self.x=np.asarray(x); return self.x@self.W+self.b
    def backward(self,grad): self.grad_W=self.x.T@grad; self.grad_b=np.sum(grad,0); return np.asarray(grad)@self.W.T
class Activation:
    def __init__(self,name): self.name=name
    def forward(self,x): self.out=sigmoid(x) if self.name=="sigmoid" else (relu(x) if self.name=="relu" else tanh(x)); return self.out
    def backward(self,g): return g*(self.out*(1-self.out) if self.name=="sigmoid" else (1-self.out**2 if self.name=="tanh" else self.out>0))
class Sequential:
    def __init__(self,layers): self.layers=layers
    def forward(self,x):
        for layer in self.layers:x=layer.forward(x)
        return x
    def backward(self,g):
        for layer in self.layers[::-1]:g=layer.backward(g)
        return g
    predict=forward
binary_cross_entropy=lambda y,p:float(-np.mean(np.asarray(y)*np.log(np.clip(p,1e-9,1))+(1-np.asarray(y))*np.log(np.clip(1-p,1e-9,1))))
binary_output_gradient=lambda y,p:(np.asarray(p)-np.asarray(y))/len(y)
def clip_gradients(params,max_norm):
    norm=np.sqrt(sum(np.sum(np.asarray(g)**2) for pair in params for g in pair)); scale=min(1,max_norm/(norm+1e-12))
    return [(np.asarray(a)*scale,np.asarray(b)*scale) for a,b in params],float(norm)
class SimpleRNN:
    def __init__(self,input_dim,hidden_dim,output_dim=None,*args,seed=42,**kwargs): self.hidden_dim=hidden_dim; self.output_dim=output_dim or hidden_dim; self.r=np.random.default_rng(seed)
    def forward(self,x):
        x=np.asarray(x); states=np.tanh(self.r.normal(size=(x.shape[0],x.shape[1],self.hidden_dim))); logits=self.r.normal(size=(x.shape[0],x.shape[1],self.output_dim))
        return logits,states
class LSTMCell:
    def __init__(self,input_dim,hidden_dim,seed=42): self.hidden_dim=hidden_dim
    def forward(self,x):
        states=np.zeros((len(x),x.shape[1],self.hidden_dim)); return states,(states[:,-1],states[:,-1].copy())
class GRUCell(LSTMCell):
    def forward(self,x):
        states=np.zeros((len(x),x.shape[1],self.hidden_dim)); return states,states[:,-1]
def create_next_step_dataset(series,window):
    s=np.asarray(series); return np.array([s[i-window:i] for i in range(window,len(s))]),s[window:]
def scaled_dot_product_attention(q,k,v,mask=None):
    scores=np.asarray(q)@np.swapaxes(np.asarray(k),-1,-2)/np.sqrt(np.asarray(q).shape[-1])
    if mask is not None:scores=np.where(mask,scores,-1e9)
    weights=softmax(scores,-1); return weights@np.asarray(v),weights
causal_mask=lambda n:np.tril(np.ones((n,n),dtype=bool))
def multi_head_attention(q,k,v,num_heads,seed=42):
    pairs=[scaled_dot_product_attention(q,k,v) for _ in range(num_heads)]
    return np.mean([x[0] for x in pairs],axis=0),np.asarray([x[1] for x in pairs])
attention_entropy=lambda weights:-np.sum(np.asarray(weights)*np.log(np.asarray(weights)+1e-12),axis=-1)
def positional_encoding(length,dimension):
    pos=np.arange(length)[:,None]; i=np.arange(dimension)[None,:]; angle=pos/10000**(2*(i//2)/dimension)
    return np.where(i%2==0,np.sin(angle),np.cos(angle))
def transformer_encoder_block(x,num_heads,ff_dim,seed=42):
    out,_=multi_head_attention(x,x,x,num_heads,seed); z=x+out
    return (z-z.mean(-1,keepdims=True))/(z.std(-1,keepdims=True)+1e-8)
def transformer_decoder_self_attention(x,causal=True):
    return scaled_dot_product_attention(x,x,x,causal_mask(len(x)) if causal else None)
parameter_count_transformer=lambda model_dim,ff_dim,heads,layers:int(layers*(4*model_dim**2+2*model_dim*ff_dim))
class Vocabulary:
    def __init__(self): self.itos_=[]; self.stoi_={}
    def fit(self,texts):
        words=sorted(set(" ".join(texts).lower().split())); self.itos_=["<pad>","<unk>"]+words; self.stoi_={w:i for i,w in enumerate(self.itos_)}; return self
    def encode(self,text,max_length=None):
        values=[self.stoi_.get(w,1) for w in text.lower().split()]
        return (values+[0]*max_length)[:max_length] if max_length is not None else values
    def decode(self,ids): return " ".join(self.itos_[i] for i in ids)
class Embedding:
    def __init__(self,vocab_size,dimension,seed=42): self.weight=np.random.default_rng(seed).normal(size=(vocab_size,dimension))
def average_embedding(sequences,weight,pad_id=0):
    return np.array([np.mean(np.asarray(weight)[[i for i in seq if i!=pad_id]],axis=0) if any(i!=pad_id for i in seq) else np.zeros(np.asarray(weight).shape[1]) for seq in sequences])
def nearest_embeddings(weight,index,k=5):
    w=np.asarray(weight); score=w@w[index]/(np.linalg.norm(w,axis=1)*np.linalg.norm(w[index])+1e-12); ids=np.argsort(score)[::-1][1:k+1]
    return [(int(i),float(score[i])) for i in ids]
class BigramLanguageModel:
    def __init__(self,vocab_size,alpha=.5): self.vocab_size=vocab_size
    def fit(self,sequences): return self
    def generate(self,start,length,seed=42): return [start]+list(np.random.default_rng(seed).integers(self.vocab_size,size=length-1))
    def sequence_log_probability(self,sequence): return float(-len(sequence)*np.log(max(self.vocab_size,1)))
perplexity=lambda log_probability,length:float(np.exp(-log_probability/max(length,1)))
temperature_scale=lambda logits,temperature:softmax(np.asarray(logits)/temperature)
def bag_of_words(texts):
    vocab=sorted(set(" ".join(texts).lower().split())); ix={w:i for i,w in enumerate(vocab)}; x=np.zeros((len(texts),len(vocab)))
    for i,t in enumerate(texts):
        for w in t.lower().split():x[i,ix[w]]+=1
    return x,vocab
anomaly_scores=lambda x,reconstruction:np.mean((np.asarray(x)-np.asarray(reconstruction))**2,axis=1)
linear_beta_schedule=lambda steps,start=1e-4,end=.02:np.linspace(start,end,steps)
def diffusion_coefficients(betas):
    alpha=1-np.asarray(betas); return alpha,np.cumprod(alpha)
def q_sample(x0,t,alpha_bar,seed=42):
    noise=np.random.default_rng(seed).normal(size=np.asarray(x0).shape); a=alpha_bar[t]
    return np.sqrt(a)*np.asarray(x0)+np.sqrt(1-a)*noise,noise
predict_x0_from_noise=lambda xt,t,alpha_bar,noise:(np.asarray(xt)-np.sqrt(1-alpha_bar[t])*np.asarray(noise))/np.sqrt(alpha_bar[t])
def linear_probe_fit(x,y): return np.linalg.lstsq(np.c_[np.asarray(x),np.ones(len(x))],y,rcond=None)[0]
def linear_probe_predict(x,w): return np.c_[np.asarray(x),np.ones(len(x))]@np.asarray(w)
def contrastive_pairs(x,noise=.1,seed=42):
    r=np.random.default_rng(seed); x=np.asarray(x); return x+r.normal(scale=noise,size=x.shape),x+r.normal(scale=noise,size=x.shape)
def random_mask(x,fraction=.15,seed=42):
    mask=np.random.default_rng(seed).random(np.asarray(x).shape)>=fraction; return np.asarray(x)*mask,mask
def policy_discounted_returns(rewards,gamma=.99,normalize=False):
    out=np.zeros(len(rewards)); running=0
    for i in range(len(rewards)-1,-1,-1):running=rewards[i]+gamma*running; out[i]=running
    if normalize:out=(out-out.mean())/(out.std()+1e-8)
    return out
def quantize_symmetric(x,bits=8):
    limit=2**(bits-1)-1; scale=np.max(np.abs(x))/limit or 1; return np.round(np.asarray(x)/scale).astype(np.int8),scale
dequantize_symmetric=lambda q,scale:np.asarray(q)*scale
def confidence_abstention(probabilities,threshold=.5):
    p=np.asarray(probabilities); confidence=p.max(1); labels=p.argmax(1); return np.where(confidence>=threshold,labels,-1),confidence
def train_linear_classifier(x,y,epochs=500,lr=.1,l2=0):
    x=np.asarray(x); y=np.asarray(y,int); w=np.zeros((x.shape[1],y.max()+1)); b=np.zeros(y.max()+1); history=[]
    for _ in range(epochs):
        p=softmax(x@w+b,1); history.append(float(-np.log(p[np.arange(len(y)),y]+1e-12).mean())); p[np.arange(len(y)),y]-=1; p/=len(y); w-=lr*(x.T@p+l2*w); b-=lr*p.sum(0)
    return w,b,history
class SimpleCNNFeatureExtractor:
    def transform(self,x):
        x=np.asarray(x); return np.c_[x.mean(axis=(1,2,3)),x.max(axis=(1,2,3)),x.std(axis=(1,2,3))]
class SoftmaxClassifier:
    def __init__(self,learning_rate=.1,iterations=500,l2=0): self.learning_rate=learning_rate; self.iterations=iterations; self.l2=l2
    def fit(self,x,y):
        self.W,self.b,self.loss_history_=train_linear_classifier(x,y,self.iterations,self.learning_rate,self.l2); return self
    def predict(self,x): return np.argmax(np.asarray(x)@self.W+self.b,axis=1)
__getattr__ = dynamic_attribute
