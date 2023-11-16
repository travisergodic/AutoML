import xgboost as xgb
import lightgbm as lgb

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from src.registry import MODEL


@MODEL.register("lr")
def build_lr(**kwargs):
    return LogisticRegression(**kwargs)

@MODEL.register("xgb")
def build_xgb(**kwargs):
    return xgb.XGBClassifier(**kwargs)

@MODEL.register("lgb")
def build_lgb(**kwargs):
    return lgb.LGBMClassifier(**kwargs)

@MODEL.register("rf")
def build_rf(**kwargs):
    return RandomForestClassifier(**kwargs)

@MODEL.register("svc")
def build_svc(**kwargs):
    return SVC(**kwargs)