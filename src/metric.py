from sklearn import metrics

from src.registry import METRIC


class BaseMetric:
    def __init__(self, **cfg):
        self.cfg=cfg

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + ", ".join([f"{k}={v}" for k, v in self.cfg.items()]) + ")"


@METRIC.register
class Accuracy(BaseMetric):
    def __call__(self, y_true, y_prob):
        return metrics.accuracy_score(y_true, y_prob.argmax(axis=1), **self.cfg)


@METRIC.register
class Recall(BaseMetric):
    def __call__(self, y_true, y_prob):
        return metrics.recall_score(y_true, y_prob.argmax(axis=1), **self.cfg)


@METRIC.register
class Precision(BaseMetric):
    def __call__(self, y_true, y_prob):
        return metrics.precision_score(y_true, y_prob.argmax(axis=1), **self.cfg)


@METRIC.register("AUC")
class AUC(BaseMetric):
    def __call__(self, y_true, y_prob):
        return metrics.roc_auc_score(y_true, y_prob[:, 1], **self.cfg)