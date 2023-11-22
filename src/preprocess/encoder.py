import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.registry import ENCODER
from .base import BaseColumnTransformer


logger=logging.getLogger(__name__)


@ENCODER.register("ordinal")
class OrdinalColumnEncoder(BaseColumnTransformer):
    TRANSFORMER_CLS=OrdinalEncoder

    def parse_cfg(self, cfg):
        if isinstance(cfg.get("categories"), list):
            cfg["categories"]=[cfg["categories"]]
        return cfg



@ENCODER.register("onehot")
class OnehotColumnEncoder(BaseColumnTransformer):
    TRANSFORMER_CLS=OneHotEncoder

    def transform(self, X, y=None):
        if not hasattr(self, "transformer"):
            return X
        onehot_df=pd.DataFrame(
            self.transformer.transform(X[[self.name]].values), 
            columns=[f"{self.name}_{category}" for category in self.transformer.categories_[0]]
        )
        return pd.concat([X.reset_index(drop=True).drop(columns=self.name), onehot_df], axis=1) 

    def parse_cfg(self, cfg):
        if isinstance(cfg.get("categories"), list):
            cfg["categories"]=[cfg["categories"]]
            cfg["sparse_output"]=False
        return cfg