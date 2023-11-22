import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from src.registry import HANDLER



@HANDLER.register("titanic_cabin")
class CabinHandler(TransformerMixin, BaseEstimator):
    def fit(self, df):
        return self

    def transform(self, df):
        df["Cabin_first_char"]=df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else x) 
        return df