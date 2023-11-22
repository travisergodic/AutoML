import numpy as np
from sklearn.impute import SimpleImputer 
from sklearn.base import TransformerMixin, BaseEstimator

from .base import BaseColumnTransformer
from src.registry import IMPUTER


@IMPUTER.register("simple")
class SimpleColumnImputer(BaseColumnTransformer): 
    TRANSFORMER_CLS=SimpleImputer
    

@IMPUTER.register("groupby")
class GroupbyColumnImputer(TransformerMixin, BaseEstimator):
    def __init__(self, name, gp_col, strategy):
        self.name=name
        self.gp_col=gp_col
        self.strategy=strategy

    def fit(self, X, y=None): 
        # if imputation strategy is set to "mean"
        if self.strategy == "mean":
            # grouping by groupby_column, find mean of null_column
            self.train_value = (
                X[X[self.name].notnull()].groupby(self.gp_col)[self.name].mean()
            )
            # calculate overall mean of null_column
            self.overall = X[X[self.name].notnull()][self.name].mean()

        # if imputation strategy is set to "median"
        elif self.strategy == "median":
            # grouping by groupby_column, find median of null_column
            self.train_value = (
                X[X[self.name].notnull()].groupby(self.gp_col)[self.name].median()
            )
            # calculate overall median of null_column
            self.overall = X[X[self.name].notnull()][self.name].median()

        # if imputation strategy is set to "most_frequent"
        elif self.strategy == "most_frequent":
            # grouping by groupby_column, find mode of null_column
            self.train_value = X[X[self.name].notnull()].groupby(self.gp_col)[self.name].agg(lambda X: X.value_counts().index[0])
            # calculate overall mode of null_column
            self.overall = X[X[self.name].notnull()][self.name].mode()[0]
        self.train_value = self.train_value.reset_index()
        return self
    
    def transform(self, X, y=None):
        # impute missing values based on train_value
        if isinstance(self.gp_col, str):
            # impute nulls with corresponding value
            X[self.name] = np.where(
                X[self.name].isnull(),
                X[self.gp_col].map(self.train_value.set_index(self.gp_col)[self.name]),
                X[self.name],
            )
            # impute any remainig nulls with overall value
            X[self.name] = X[self.name].fillna(value=self.overall)
        return X[self.name]