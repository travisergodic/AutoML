from sklearn.base import BaseEstimator, TransformerMixin


class BaseColumnTransformer(TransformerMixin, BaseEstimator):
    TRANSFORMER_CLS=None
    def __init__(self, name, **cfg):
        self.name=name
        self.cfg=self.parse_cfg(cfg)

    def fit(self, X, y=None):
        if self.name in X:
            self.transformer=self.__class__.TRANSFORMER_CLS(**self.cfg).fit(X[[self.name]].values)
        return self
    
    def transform(self, X, y=None):
        if not hasattr(self, "transformer"):
            return X
        X[self.name]=self.transformer.transform(X[[self.name]].values).squeeze(axis=-1) 
        return X 
    
    def parse_cfg(self, cfg):
        return cfg
    