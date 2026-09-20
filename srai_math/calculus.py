import numpy as np
from srai_compat import dynamic_attribute

def derivative(f, x, h=1e-5, method="central"):
    if method=="forward": return (f(x+h)-f(x))/h
    if method=="backward": return (f(x)-f(x-h))/h
    return (f(x+h)-f(x-h))/(2*h)
second_derivative = lambda f, x, h=1e-4: (f(x + h) - 2 * f(x) + f(x - h)) / h**2
gradient = lambda f, x, h=1e-5: np.asarray([(f(np.asarray(x) + np.eye(len(x))[i]*h)-f(np.asarray(x)-np.eye(len(x))[i]*h))/(2*h) for i in range(len(x))])
directional_derivative = lambda f, x, d: float(np.dot(gradient(f, x), np.asarray(d)/np.linalg.norm(d)))
trapezoidal_rule = lambda f, a, b, n=1000: np.trapezoid(f(np.linspace(a,b,n+1)), np.linspace(a,b,n+1))
midpoint_rule = lambda f, a, b, n=1000: (b-a)/n*np.sum(f(a+(np.arange(n)+.5)*(b-a)/n))
left_riemann = lambda f, a, b, n=1000: (b-a)/n*np.sum(f(np.linspace(a,b,n,endpoint=False)))
right_riemann = lambda f, a, b, n=1000: (b-a)/n*np.sum(f(np.linspace(a,b,n+1)[1:]))

def numerical_limit(f,x,h=1e-6): return (f(x-h)+f(x+h))/2
class Dual:
    def __init__(self,value,derivative=0): self.value=value; self.derivative=derivative
    def __add__(self,o):
        o=o if isinstance(o,Dual) else Dual(o)
        return Dual(self.value+o.value,self.derivative+o.derivative)
    __radd__=__add__
    def __mul__(self,o):
        o=o if isinstance(o,Dual) else Dual(o)
        return Dual(self.value*o.value,self.derivative*o.value+self.value*o.derivative)
    __rmul__=__mul__
    def __pow__(self,power): return Dual(self.value**power,power*self.value**(power-1)*self.derivative)
    def __sub__(self,o): return self+(-1)*o
    def __rsub__(self,o): return o+(-1)*self
    def __repr__(self): return f"Dual(value={self.value}, derivative={self.derivative})"
def approach_sequence(point,side="both",n=10):
    h=10.0**(-np.arange(1,n+1))
    return point-h if side=="left" else (point+h if side=="right" else (point-h,point+h))
def one_sided_limits(f,x,h=1e-6): return f(x-h),f(x+h)
def removable_extension(f,point,value): return lambda x:value if np.isclose(x,point) else f(x)
def is_continuous_at(f,x,tol=1e-5): return np.isclose(numerical_limit(f,x),f(x),rtol=tol,atol=tol)
def epsilon_delta_check(f,point=None,limit_value=None,epsilon=.01,delta=.001,**kwargs):
    return all(abs(f(v)-limit_value)<epsilon for v in (point-delta/2,point+delta/2))
def jacobian(f,x,h=1e-5):
    x=np.asarray(x,float)
    return np.column_stack([(np.asarray(f(x+np.eye(len(x))[i]*h))-np.asarray(f(x-np.eye(len(x))[i]*h)))/(2*h) for i in range(len(x))])
def hessian(f,x,h=1e-4):
    x=np.asarray(x,float); n=len(x); out=np.empty((n,n))
    for i in range(n):
        for j in range(n):
            ei=np.eye(n)[i]*h; ej=np.eye(n)[j]*h
            out[i,j]=(f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej))/(4*h*h)
    return out
def taylor_first_order(*args):
    if callable(args[0]):
        f,x,x0=args; return f(x0)+gradient(f,np.asarray(x0,float))@(np.asarray(x)-np.asarray(x0))
    f_center,g_center,x_target,x_center=args
    return f_center+np.asarray(g_center)@(np.asarray(x_target)-np.asarray(x_center))
dual_sin=lambda x:Dual(np.sin(x.value),np.cos(x.value)*x.derivative)
dual_cos=lambda x:Dual(np.cos(x.value),-np.sin(x.value)*x.derivative)
dual_exp=lambda x:Dual(np.exp(x.value),np.exp(x.value)*x.derivative)
dual_log=lambda x:Dual(np.log(x.value),x.derivative/x.value)
def autodiff_derivative(f,x):
    y=f(Dual(x,1.0))
    return y.value,y.derivative
def autodiff_gradient(f,x):
    x=np.asarray(x,float); result=[]; value=None
    for i in range(len(x)):
        y=f(np.array([Dual(v,1.0 if i==j else 0.0) for j,v in enumerate(x)],dtype=object))
        value=y.value
        result.append(y.derivative)
    return value,np.asarray(result)
__getattr__ = dynamic_attribute
