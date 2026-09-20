import numpy as np
def tropical_livestock_units(cattle=0,sheep=0,goats=0,camels=0,poultry=0):
    return .7*np.asarray(cattle,float)+.1*np.asarray(sheep,float)+.1*np.asarray(goats,float)+np.asarray(camels,float)+.01*np.asarray(poultry,float)
def mortality_rate(deaths,opening_stock,births=0,purchases=0):
    d=np.asarray(opening_stock,float)+np.asarray(births,float)+np.asarray(purchases,float)
    return np.divide(np.asarray(deaths,float),d,out=np.zeros_like(d),where=d>0)
def offtake_rate(sales,slaughter,opening_stock):
    n=np.asarray(sales,float)+np.asarray(slaughter,float); d=np.asarray(opening_stock,float)
    return np.divide(n,d,out=np.zeros_like(n),where=d>0)
def herd_growth(closing_stock,opening_stock):
    o=np.asarray(opening_stock,float)
    return np.divide(np.asarray(closing_stock,float)-o,o,out=np.zeros_like(o),where=o>0)
def livestock_summary(df):
    cols=[c for c in ["cattle","sheep","goats","camels","poultry"] if c in df.columns]
    out={c:float(df[c].sum()) for c in cols}
    if cols: out["holdings_with_livestock"]=int((df[cols].sum(axis=1)>0).sum())
    return out
