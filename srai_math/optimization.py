import numpy as np
from srai_compat import dynamic_attribute

quadratic_value = lambda x,q,b=None,c=0: .5*np.asarray(x)@np.asarray(q)@np.asarray(x)+(0 if b is None else np.asarray(b)@np.asarray(x))+c
quadratic_gradient = lambda x,q,b=None: np.asarray(q)@np.asarray(x)+(0 if b is None else np.asarray(b))
project_box = lambda x,lower,upper: np.clip(x,lower,upper)
is_convex_quadratic = lambda q: bool(np.all(np.linalg.eigvalsh(q)>=-1e-10))
constraint_violation = lambda x,lower,upper: float(np.sum(np.maximum(lower-np.asarray(x),0)+np.maximum(np.asarray(x)-upper,0)))

def gradient_descent(f,g,x0,learning_rate=.01,max_iter=1000,tol=1e-8):
    x=np.asarray(x0,float).copy(); history=[float(f(x))]
    for i in range(max_iter):
        step=learning_rate*np.asarray(g(x)); x-=step; history.append(float(f(x)))
        if np.linalg.norm(step)<tol: break
    return x,history,i+1
def project_simplex(v,z=1.0):
    v=np.asarray(v,float); u=np.sort(v)[::-1]; css=np.cumsum(u)-z
    rho=np.nonzero(u-css/(np.arange(len(v))+1)>0)[0][-1]
    return np.maximum(v-css[rho]/(rho+1),0)
def projected_gradient_descent(f,g,x0,projection,learning_rate=.01,max_iter=1000,tol=1e-8):
    x=np.asarray(x0,float); history=[float(f(x))]
    for i in range(max_iter):
        new=np.asarray(projection(x-learning_rate*np.asarray(g(x))))
        history.append(float(f(new)))
        if np.linalg.norm(new-x)<tol: x=new; break
        x=new
    return x,history,i+1
def momentum_descent(f,g,x0,learning_rate=.01,momentum=.9,max_iter=1000,tol=1e-8):
    x=np.asarray(x0,float); v=np.zeros_like(x); h=[float(f(x))]
    for i in range(max_iter):
        v=momentum*v+np.asarray(g(x)); step=learning_rate*v; x-=step; h.append(float(f(x)))
        if np.linalg.norm(step)<tol: break
    return x,h
def rmsprop(f,g,x0,learning_rate=.001,decay=.9,max_iter=1000,tol=1e-8):
    x=np.asarray(x0,float); avg=np.zeros_like(x); h=[float(f(x))]
    for i in range(max_iter):
        grad=np.asarray(g(x)); avg=decay*avg+(1-decay)*grad**2
        step=learning_rate*grad/(np.sqrt(avg)+1e-8); x-=step; h.append(float(f(x)))
        if np.linalg.norm(step)<tol: break
    return x,h
def adam(f,g,x0,learning_rate=.001,beta1=.9,beta2=.999,max_iter=1000,tol=1e-8):
    x=np.asarray(x0,float); m=np.zeros_like(x); v=np.zeros_like(x); h=[float(f(x))]
    for i in range(1,max_iter+1):
        grad=np.asarray(g(x)); m=beta1*m+(1-beta1)*grad; v=beta2*v+(1-beta2)*grad**2
        step=learning_rate*(m/(1-beta1**i))/(np.sqrt(v/(1-beta2**i))+1e-8); x-=step; h.append(float(f(x)))
        if np.linalg.norm(step)<tol: break
    return x,h
__getattr__ = dynamic_attribute
