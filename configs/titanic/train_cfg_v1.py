list_of_handler_cfg=["titanic_cabin"]

list_of_imputer_cfg=[
    dict(name="Cabin_first_char", type="simple", strategy="most_frequent"),
    dict(name="Age", type="simple", strategy="median"), 
    dict(name="Fare", type="simple", strategy="median"),
    dict(name="Embarked", type="simple", strategy="most_frequent")
]

list_of_encoder_cfg=[
    dict(name="Cabin_first_char", type="onehot", categories="auto", sparse_output=False), 
    dict(name="Pclass", type="ordinal", categories="auto"),
    dict(name="Sex", type="ordinal", categories="auto"),
    dict(name="Embarked", type="ordinal", categories="auto")
]

list_of_normalizer_cfg=[]

used_cols=["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Cabin_first_char", "Pclass"]

model_cfg={
    "type": "xgb", "learning_rate": 0.02, "n_estimators": 300, 
    "objective": "binary:logistic", "random_state": 42, 
    "enable_categorical": True, "verbosity": 1
}

metrics=["accuracy", "recall", "precision"]