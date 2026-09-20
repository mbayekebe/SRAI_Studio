import numpy as np
def stock_flow_simulation(initial_stock,inflows,outflows,steps,dt=1.0):
    stock=float(initial_stock); history=[stock]
    for t in range(steps):
        stock=stock+dt*(inflows(t,stock)-outflows(t,stock)); history.append(stock)
    return np.asarray(history)
def exponential_growth(initial,growth_rate,steps):
    return stock_flow_simulation(initial,lambda t,s:growth_rate*s,lambda t,s:0,steps)
def carrying_capacity_growth(initial,growth_rate,capacity,steps):
    return stock_flow_simulation(initial,lambda t,s:growth_rate*s*(1-s/capacity),lambda t,s:0,steps)
def feedback_loop_gain(coefficients):
    return float(np.prod(np.asarray(coefficients,float)))
def equilibrium(inflow_function,outflow_function,grid):
    values=[abs(inflow_function(x)-outflow_function(x)) for x in grid]
    return float(grid[int(np.argmin(values))])
