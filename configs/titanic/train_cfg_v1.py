list_of_handler_cfg=["titanic_cabin"]

list_of_imputer_cfg=[
    dict(name="Cabin_first_char", type="simple", strategy="most_frequent"),
    dict(name="Age", type="single_col_groupby", gp_col="Sex", strategy="median"), 
    # dict(name="Age", type="simple", strategy="median"), 
    dict(name="Fare", type="simple", strategy="median"),
    dict(name="Embarked", type="simple", strategy="most_frequent")
]

list_of_encoder_cfg=[
    dict(name="Cabin_first_char", type="onehot", categories="auto"), 
    dict(name="Pclass", type="ordinal", categories="auto"),
    dict(name="Sex", type="ordinal", categories="auto"),
    dict(name="Embarked", type="ordinal", categories="auto")
]

list_of_normalizer_cfg=[
    dict(name="Age", type="standard")
]


used_cols=["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Cabin_first_char", "Pclass"]

model_cfg={
    "type": "xgb", "learning_rate": 0.02, "n_estimators": 300, 
    "objective": "binary:logistic", "random_state": 42, 
    "enable_categorical": True, "verbosity": 1
}

param_grid={
    "n_estimators": [200, 300],
    "min_child_weight": [1, 5, 10],
    "gamma": [0.5, 1, 1.5, 2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "max_depth": [4, 5]
}

search_cfg={
    "type": "grid_search",
    "param_grid": param_grid, 
    "scoring": "accuracy",
    "cv": 3,
    "return_train_score": False
}


list_of_metric_cfg=[
    dict(type="Accuracy"), dict(type="Recall"), dict(type="Precision"), dict(type="AUC", average="micro")
]