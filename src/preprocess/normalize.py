import logging

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.registry import NORMALIZER
from .base import BaseColumnTransformer


@NORMALIZER.register("minmax")
class MinMaxNormalizer(BaseColumnTransformer):
    TRANSFORMER_CLS=MinMaxScaler


@NORMALIZER.register("standard")
class StandardNormalizer(BaseColumnTransformer):
    TRANSFORMER_CLS=StandardScaler