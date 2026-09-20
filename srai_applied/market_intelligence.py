"""Agricultural market-price intelligence."""
from __future__ import annotations
import numpy as np
import pandas as pd

def price_index(prices,base_periods=3):
    p=pd.Series(prices,dtype=float)
    base=p.iloc[:base_periods].mean()
    return 100*p/base

def price_change(prices,periods=1):
    return pd.Series(prices,dtype=float).pct_change(periods)

def rolling_volatility(prices,window=4):
    returns=pd.Series(prices,dtype=float).pct_change()
    return returns.rolling(window).std()

def market_spread(producer_price,retail_price):
    producer=np.asarray(producer_price,float)
    retail=np.asarray(retail_price,float)
    return retail-producer

def market_integration_correlation(series_a,series_b):
    a=np.asarray(series_a,float); b=np.asarray(series_b,float)
    return float(np.corrcoef(a,b)[0,1])

def shock_flags(prices,z_threshold=2.0):
    p=pd.Series(prices,dtype=float)
    changes=p.pct_change()
    z=(changes-changes.mean())/changes.std(ddof=0)
    return z.abs()>z_threshold

def food_basket_index(price_frame,weights):
    frame=pd.DataFrame(price_frame)
    w=pd.Series(weights,dtype=float)
    normalized=frame/frame.iloc[0]
    return 100*normalized.mul(w,axis=1).sum(axis=1)/w.sum()
