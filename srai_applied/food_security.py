import numpy as np
def food_consumption_score(staples,pulses,vegetables,fruit,meat,milk,sugar,oil):
    return 2*min(staples,7)+3*min(pulses,7)+min(vegetables,7)+min(fruit,7)+4*min(meat,7)+4*min(milk,7)+.5*min(sugar,7)+.5*min(oil,7)
def fcs_category(score):
    return "poor" if score<=21 else "borderline" if score<=35 else "acceptable"
def reduced_coping_strategy_index(less_preferred,borrow_food,limit_portions,restrict_adults,reduce_meals):
    return np.asarray(less_preferred,float)+2*np.asarray(borrow_food,float)+np.asarray(limit_portions,float)+3*np.asarray(restrict_adults,float)+np.asarray(reduce_meals,float)
def household_food_expenditure_share(food_expenditure,total_expenditure):
    f=np.asarray(food_expenditure,float); t=np.asarray(total_expenditure,float)
    return np.divide(f,t,out=np.zeros_like(f),where=t>0)
def food_security_class(fcs,rcsi,food_share):
    if fcs<=21 or rcsi>=20 or food_share>=.75: return "severely_insecure"
    if fcs<=35 or rcsi>=10 or food_share>=.65: return "moderately_insecure"
    return "food_secure"
def prevalence_by_group(df,group_col,class_col):
    return df.groupby(group_col)[class_col].value_counts(normalize=True).rename("share").reset_index()
