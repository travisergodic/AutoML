import os
import logging
from pathlib import Path


import joblib
import numpy as np
import pandas as pd


from .metric import METRIC
from src.utils import get_pretty_table, save_json, load_json
from src.utils import get_feature_importance_df


logger = logging.getLogger(__name__)

class InferenceMixin:
    def predict(self, X):
        X=self.preprocessor.transform(X[self.used_cols])
        return self.model.predict(X)
  
    def predict_proba(self, X):
        X=self.preprocessor.transform(X[self.used_cols])
        return self.model.predict_proba(X)


    def get_pred_result(self, df):
        prob=self.predict_proba(df)
        df_prob=pd.DataFrame(
            prob, columns=[f"prob_{i}" for i in range(prob.shape[1])]
        )
        return pd.concat([df.reset_index(drop=True), df_prob], axis=1)   


    def evaluate(self, df):
        if not self.metrics:
            raise ValueError("Have not specify metrics for evaluation.")
      
        if len(df) == 0:
            logger.warning("Input dataframe does not have any element.")
            return {}
  
        return {
            metric: METRIC.build(type=metric, y_true=df[self.label_col].values, y_prob=self.predict_proba(df)) \
                for metric in self.metrics
        }


class Predictor(InferenceMixin):
    @classmethod
    def build_from_exp_dir(cls, exp_dir, label_col=None, metrics=None):
        model=joblib.load(os.path.join(exp_dir, "model.pkl"))
        preprocessor=joblib.load(os.path.join(exp_dir, "preprocessor.pkl"))
        used_cols=load_json(os.path.join(exp_dir, "use_cols.json"))
        return cls(model, preprocessor, used_cols, label_col, metrics)
      
    def __init__(self, model, preprocessor, used_cols, label_col=None, metrics=None):
        self.model = model
        self.preprocessor=preprocessor
        self.used_cols=used_cols
        self.label_col=label_col
        self.metrics=metrics


class Trainer(InferenceMixin):   
    def __init__(self, model, preprocessor, used_cols, label_col, metrics=None):
        self.model=model
        self.preprocessor=preprocessor
        self.used_cols=used_cols
        self.label_col=label_col
        self.metrics=metrics
      
    def fit(self, df_train):   
        X_train=self.preprocessor.fit_transform(df_train[self.used_cols])
        y_train=df_train[self.label_col].values
        self._features=X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        return self.model
  
    def train(self, df_train, df_test=None):
        model=self.fit(df_train)
        self.performance_df=pd.DataFrame.from_records(
            [self.evaluate(df_train), self.evaluate(df_test)], index=["train", "test"]
        ).reset_index(drop=False).dropna()
        logger.info(get_pretty_table(self.performance_df, "model performance"))


        # get pred result df
        df_train_result=self.get_pred_result(df_train)
        df_test_result=self.get_pred_result(df_test)
        df_train_result["is_train"]=True
        df_test_result["is_train"]=False
        self.pred_result_df=pd.concat([df_train_result, df_test_result], axis=0)
        return model  


    def get_result_df(self, df_train=None, df_test=None):
        # calculate prob
        prob=np.concatenate([self.predict_proba(df_train), self.predict_proba(df_test)], axis=0)
        df_prob=pd.DataFrame(prob, columns=[f"prob_{i}" for i in range(prob.shape[1])])
        df_res=pd.concat(
            [df_train.reset_index(drop=True), df_test.reset_index(drop=True), df_prob], axis=1
        )
        df_res["is_train"]= [True] * len(df_train) + [False] * len(df_test)
        return df_res


    def save(self, checkpoint_dir):       
        model_ckpt=os.path.join(checkpoint_dir, "model.pkl")
        preprocessor_ckpt=os.path.join(checkpoint_dir, "preprocessor.pkl")
        metric_path=os.path.join(checkpoint_dir, "metric.csv")
        pred_result_path=os.path.join(checkpoint_dir, "pred_result.csv")
        use_cols_path=os.path.join(checkpoint_dir, "use_cols.json")
        features_path=os.path.join(checkpoint_dir, "features.json")
      
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        model=self.model.best_estimator_ if hasattr(self.model, "best_estimator_") else self.model
        joblib.dump(model, model_ckpt)
        logger.info(f"Save model at {model_ckpt}")
      
        joblib.dump(self.preprocessor, preprocessor_ckpt)
        logger.info(f"Save preprocessor at {preprocessor_ckpt}")

        self.performance_df.to_csv(metric_path, index=False)
        logger.info(f"Save metrics at {metric_path}")

        self.pred_result_df.to_csv(pred_result_path, index=False)
        logger.info(f"Save prediction result at {pred_result_path}")

        save_json(self.used_cols, use_cols_path)
        logger.info(f"Save use columns at {use_cols_path}")
      
        save_json(self._features, features_path)
        logger.info(f"Save features at {features_path}")
       
        try: 
            importance_path=os.path.join(checkpoint_dir, "importance.csv")
            get_feature_importance_df(model, self._features).to_csv(importance_path, index=False)
            logger.info(f"Save feature importance table at {importance_path}")
        except Exception as e:
            logger.warning(f"Fail to get feature importance of {self.model}.")