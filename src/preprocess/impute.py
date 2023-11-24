import logging

import numpy as np
from sklearn.impute import SimpleImputer 

from .base import BaseColumnTransformer
from src.registry import IMPUTER
from src.utils import get_pretty_table


logger=logging.getLogger(__name__)


@IMPUTER.register("simple")
class SimpleColumnImputer(BaseColumnTransformer): 
    TRANSFORMER_CLS=SimpleImputer
    

@IMPUTER.register("single_col_groupby")
class GroupbySingleColumnImputer(BaseColumnTransformer):
    def __init__(self, name, gp_col, strategy):
        self.name=name
        self.gp_col=gp_col
        self.strategy=strategy

    def fit(self, X, y=None): 
        # if imputation strategy is set to "mean"
        if self.strategy == "mean":        
            self.train_value = X[X[self.name].notnull()].groupby(self.gp_col)[self.name].mean()
            self.overall = X[X[self.name].notnull()][self.name].mean()

        # if imputation strategy is set to "median"
        elif self.strategy == "median":
            self.train_value = X[X[self.name].notnull()].groupby(self.gp_col)[self.name].median()
            self.overall = X[X[self.name].notnull()][self.name].median()

        # if imputation strategy is set to "most_frequent"
        elif self.strategy == "most_frequent":
            self.train_value = X[X[self.name].notnull()].groupby(self.gp_col)[self.name].agg(lambda X: X.value_counts().index[0])
            self.overall = X[X[self.name].notnull()][self.name].mode()[0]

        logger.info(get_pretty_table(self.train_value, title=f"{self.name} column train_value"))
        logger.info(f"{self.name} column has overall value {self.overall}")
        return self
    
    def transform(self, X, y=None):
        X[self.name] = np.where(X[self.name].isnull(), X[self.gp_col].map(self.train_value), X[self.name])
        X[self.name] = X[self.name].fillna(value=self.overall)
        return X
    

@IMPUTER.register("multi_col_groupby")
class GroupbyMultiColumnImputer(BaseColumnTransformer):
    def __init__(self, name, gp_cols, strategy):
        self.name=name
        self.gp_cols=gp_cols
        self.strategy=strategy

    def fit(self, X, y=None): 
        # if imputation strategy is set to "mean"
        if self.strategy == "mean":        
            self.train_value = X[X[self.name].notnull()].groupby(self.gp_cols)[self.name].mean()
            self.overall = X[X[self.name].notnull()][self.name].mean()

        # if imputation strategy is set to "median"
        elif self.strategy == "median":
            self.train_value = X[X[self.name].notnull()].groupby(self.gp_cols)[self.name].median()
            self.overall = X[X[self.name].notnull()][self.name].median()

        # if imputation strategy is set to "most_frequent"
        elif self.strategy == "most_frequent":
            self.train_value = X[X[self.name].notnull()].groupby(self.gp_cols)[self.name].agg(lambda X: X.value_counts().index[0])
            self.overall = X[X[self.name].notnull()][self.name].mode()[0]
        return self
    
    def transform(self, X, y=None):
        X_replace=X.apply(lambda row: tuple(row[col] for col in self.gp_cols), axis=1).map(self.train_value)
        X[self.name]=np.where(X[self.name].isnull(), X_replace, X[self.name])
        X[self.name]=X[self.name].fillna(value=self.overall)
        return X