import scipy.stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.registry import SEARCH

def parse_param_dist(param_dist):
    def parse_param_dist_dict(param_dist_dict):
        res_dict={}
        for name, dist in param_dist_dict.items():
            if isinstance(dist, list):
                res_dict[name]=dist               
            elif isinstance(dist, str) and dist.startswith("scipy"): 
                res_dict[name]=eval(dist)
            else:
                raise ValueError()
        return res_dict
    
    if isinstance(param_dist, list):
        return [parse_param_dist_dict(ele) for ele in param_dist]
    if isinstance(param_dist, dict):
        return parse_param_dist_dict(param_dist) 
    

@SEARCH.register("grid_search")
def build_grid_search(model, **kwargs):
    return GridSearchCV(model, **kwargs)

@SEARCH.register("random_search")
def build_random_search(model, **kwargs):
    param_dist = kwargs.pop("param_distributions")
    param_dist=parse_param_dist(param_dist) 
    return RandomizedSearchCV(model=model, param_distributions=param_dist, **kwargs)  