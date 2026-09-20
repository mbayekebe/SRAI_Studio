"""First-order optimization algorithms."""
from __future__ import annotations
import numpy as np

def gradient_descent(function,gradient,x0,learning_rate=0.1,max_iter=1000,tol=1e-8,projection=None):
    x=np.asarray(x0,dtype=float).copy()
    history=[float(function(x))]
    for i in range(1,max_iter+1):
        g=np.asarray(gradient(x),dtype=float)
        if np.linalg.norm(g)<=tol:
            return x,history,i-1
        x=x-learning_rate*g
        if projection is not None:
            x=np.asarray(projection(x),dtype=float)
        history.append(float(function(x)))
    return x,history,max_iter

def backtracking_line_search(function,gradient,x,direction,alpha=1.0,beta=0.5,c=1e-4):
    x=np.asarray(x,dtype=float); d=np.asarray(direction,dtype=float)
    g=np.asarray(gradient(x),dtype=float)
    fx=float(function(x))
    while function(x+alpha*d)>fx+c*alpha*(g@d):
        alpha*=beta
        if alpha<1e-16:
            break
    return float(alpha)

def momentum_descent(function,gradient,x0,learning_rate=0.05,momentum=0.9,max_iter=500):
    x=np.asarray(x0,dtype=float).copy()
    v=np.zeros_like(x)
    history=[float(function(x))]
    for _ in range(max_iter):
        v=momentum*v-learning_rate*np.asarray(gradient(x),dtype=float)
        x=x+v
        history.append(float(function(x)))
    return x,history

def rmsprop(function,gradient,x0,learning_rate=0.01,decay=0.9,epsilon=1e-8,max_iter=500):
    x=np.asarray(x0,dtype=float).copy()
    avg=np.zeros_like(x)
    history=[float(function(x))]
    for _ in range(max_iter):
        g=np.asarray(gradient(x),dtype=float)
        avg=decay*avg+(1-decay)*g*g
        x=x-learning_rate*g/(np.sqrt(avg)+epsilon)
        history.append(float(function(x)))
    return x,history

def adam(function,gradient,x0,learning_rate=0.05,beta1=0.9,beta2=0.999,epsilon=1e-8,max_iter=500):
    x=np.asarray(x0,dtype=float).copy()
    m=np.zeros_like(x); v=np.zeros_like(x)
    history=[float(function(x))]
    for t in range(1,max_iter+1):
        g=np.asarray(gradient(x),dtype=float)
        m=beta1*m+(1-beta1)*g
        v=beta2*v+(1-beta2)*g*g
        mh=m/(1-beta1**t)
        vh=v/(1-beta2**t)
        x=x-learning_rate*mh/(np.sqrt(vh)+epsilon)
        history.append(float(function(x)))
    return x,history

def projected_gradient_descent(function,gradient,x0,projection,learning_rate=0.1,max_iter=1000,tol=1e-8):
    return gradient_descent(function,gradient,x0,learning_rate,max_iter,tol,projection)
