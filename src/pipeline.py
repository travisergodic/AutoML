from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from src.handler import HANDLER
from src.preprocess.encoder import ENCODER
from src.preprocess.impute import IMPUTER
from src.preprocess.normalize import NORMALIZER


class IdentityPipeline(TransformerMixin, BaseEstimator):
    def fit(self, df):
        return self
    
    def transform(self, df):
        return df


def build_handler_pipeline(list_of_handler=None):
    if (list_of_handler is None) or (not list_of_handler):
        return IdentityPipeline()
    return Pipeline([(ele, HANDLER.build(type=ele)) for ele in list_of_handler], verbose=True)


def build_encoder_pipeline(list_of_encoder_cfg=None):
    if (list_of_encoder_cfg is None) or (not list_of_encoder_cfg):
        return IdentityPipeline()
    return Pipeline(
        [(cfg["name"], ENCODER.build(**cfg)) for cfg in list_of_encoder_cfg], verbose=True
    )


def build_imputer_pipeline(list_of_imputer_cfg=None):
    if (list_of_imputer_cfg is None) or (not list_of_imputer_cfg):
        return IdentityPipeline()
    return Pipeline(
        [(cfg["name"], IMPUTER.build(**cfg)) for cfg in list_of_imputer_cfg], verbose=True
    )


def build_normalizer_pipeline(list_of_normalizer_cfg=None):
    if (list_of_normalizer_cfg is None) or (not list_of_normalizer_cfg):
        return IdentityPipeline()
    return Pipeline(
        [(cfg["name"], NORMALIZER.build(**cfg)) for cfg in list_of_normalizer_cfg], verbose=True
    )