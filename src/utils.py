import os
import sys
import logging
import importlib

import json
import pandas as pd
from rich.table import Table
from rich.console import Console


logger=logging.getLogger(__name__)


def get_feature_importance_df(model, features):
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{model} does not have attribute: feature_importance_.")
    return pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values(by=["importance"], ascending=False)


def get_cfg_by_file(cfg_file):
    try:
        sys.path.append(os.path.dirname(cfg_file))
        current_cfg = importlib.import_module(os.path.basename(cfg_file).split(".")[0])
        logger.info(f'Import {cfg_file} successfully!')
    except Exception:
        raise ImportError(f'Fail to import {cfg_file}')
    return current_cfg


def get_pretty_table(data, title):
    if isinstance(data, pd.Series):
        data=data.to_frame().reset_index()
    
    elif not isinstance(data, pd.DataFrame):
        raise ValueError()

    table=Table(title=title)
    for col in data:
        table.add_column(col, justify="left", style="cyan")
        
    for _, row in data.iterrows():
        table.add_row(*[str(row[col]) for col in data])

    console = Console()
    with console.capture() as capture:
        console.print(table, end='')
    return capture.get()


def save_json(obj, path, indent=4):
    f = open(path, "w") 
    json.dump(obj, f, indent=indent) 

def load_json(path):
    f = open(path, "r")
    return json.load(f) 
