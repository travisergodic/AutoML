from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from src.encoder import ENCODER
from src.impute import IMPUTER
from src.normalize import NORMALIZER


class IdentityPipeline(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        return df


def build_encoder_pipeline(list_of_encoder_cfg):
    if (list_of_encoder_cfg is None) or (len(list_of_encoder_cfg) == 0):
        return IdentityPipeline()
    return Pipeline(
        [(encoder_cfg["name"], ENCODER.build(encoder_cfg)) for encoder_cfg in list_of_encoder_cfg], verbose=True
    )


def build_imputer_pipeline(list_of_imputer_cfg):
    if (list_of_imputer_cfg is None) or (len(list_of_imputer_cfg) == 0):
        return IdentityPipeline()
    return Pipeline(
        [(encoder_cfg["name"], IMPUTER.build(encoder_cfg)) for encoder_cfg in list_of_imputer_cfg], verbose=True
    )


def build_normalizer_pipeline(list_of_normalizer_cfg):
    if (list_of_normalizer_cfg is None) or (len(list_of_normalizer_cfg) == 0):
        return IdentityPipeline()
    return Pipeline(
        [(encoder_cfg["name"], NORMALIZER.build(encoder_cfg)) for encoder_cfg in list_of_normalizer_cfg], verbose=True
    )