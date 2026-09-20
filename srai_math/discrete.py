from itertools import product
from srai_compat import dynamic_attribute

union=lambda a,b:set(a)|set(b)
intersection=lambda a,b:set(a)&set(b)
difference=lambda a,b:set(a)-set(b)
complement=lambda a,universe:set(universe)-set(a)
cartesian_product=lambda a,b:set(product(a,b))
implies=lambda p,q:(not p) or q
relation_domain=lambda r:{a for a,_ in r}
relation_range=lambda r:{b for _,b in r}
is_function=lambda r: len({a for a,_ in r})==len(r)
truth_table_2=lambda f:[{"P":p,"Q":q,"result":f(p,q)} for p in (True,False) for q in (True,False)]
evaluate_predicate=lambda items,predicate:[item for item in items if predicate(item)]
__getattr__=dynamic_attribute
