import logging

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.registry import ENCODER


logger=logging.getLogger(__name__)


@ENCODER.register("ordinal")
class OrdinalColumnEncoder(TransformerMixin):
    def __init__(self, name, **cfg):
        self.name=name
        self.cfg=cfg

    def fit(self, df):
        if self.name in df: 
            self.encoder=OrdinalEncoder(**self.cfg).fit(df[[self.name]])
        return self
    
    def transform(self, df):
        if not hasattr(self, "encoder"):
            return df
        df[self.name]=self.encoder.transform(df[[self.name]]).squeeze(axis=-1) 
        return df 


@ENCODER.register("onehot")
class OnehotColumnEncoder(TransformerMixin):
    def __init__(self, name, cfg):
        self.name=name
        self.cfg=cfg

    def fit(self, df):
        if self.name in df: 
            self.encoder=OneHotEncoder(**self.cfg).fit(df[[self.name]])
        return self
    
    def transform(self, df):
        if not hasattr(self, "encoder"):
            return df
        onehot_vec=self.encoder.transform(df[[self.name]]).squeeze(axis=-1) 
        onehot_df=pd.DataFrame(
            onehot_vec, columns=[f"{self.name}_{i}" for i in range(onehot_vec.shape[-1])]
        )
        return pd.concat(
            [df.reset_index(drop=True), onehot_df], axis=1
        ).drop(columns=self.name)  