from sklearn import metrics

from src.registry import METRIC


@METRIC.register("accuracy")
def accuracy_score(y_true, y_prob):    
    return metrics.accuracy_score(y_true, y_prob.argmax(axis=1))

@METRIC.register("recall")
def recall_score(y_true, y_prob):    
    return metrics.recall_score(y_true, y_prob.argmax(axis=1))

@METRIC.register("precision")
def precision_score(y_true, y_prob):    
    return metrics.precision_score(y_true, y_prob.argmax(axis=1))

@METRIC.register("auc")
def auc_score(y_true, y_prob):
    return metrics.roc_auc_score(y_true, y_prob[:, 1])