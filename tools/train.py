import os
import sys
import logging
import argparse
sys.path.insert(0, os.getcwd())

import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils import get_cfg_by_file
from src.logger_helper import setup_logger
from src.trainer import Trainer
from src.model import MODEL
from src.search import SEARCH
from src.utils import get_feature_importance_df
from src.pipeline import build_encoder_pipeline, build_imputer_pipeline, build_normalizer_pipeline


logger = setup_logger(level=logging.INFO)


def main():
    df = pd.read_csv(args.csv_path, low_memory=False)
    logger.info(f"Read csv file from {args.csv_path}.")

    # build preprocessor
    encoder_pipeline=build_encoder_pipeline(config.list_of_encoder_cfg)
    imputer_pipeline=build_imputer_pipeline(config.list_of_imputer_cfg)
    normalizer_pipeline=build_normalizer_pipeline(config.list_of_normalizer_cfg)
    preprocessor=Pipeline(
        ("encoder", encoder_pipeline), 
        ("imputer", imputer_pipeline), 
        ("normalizer", normalizer_pipeline), verbose=True
    )

    # build model
    model=MODEL.build(**config.model_cfg)

    # build search
    if args.search and config.search_cfg:
        search = SEARCH.build(model=model, **config.search_cfg)
  
    # build trainer
    trainer = Trainer(
        (model if not args.search else search),
        preprocessor, config.used_cols, args.label_cols, 
        metrics=config.metrics
    )

    # train model
    trainer.train(df[df[args.split_col]], df[~df[args.split_col]])

    # save checkpoints
    checkpoint_dir=f"./checkpoints/{args.exp_name}"
    trainer.save(checkpoint_dir)
    model=(model if not args.search else search.best_estimator_)
    if hasattr(model, "feature_importance_"): 
        get_feature_importance_df(model, trainer.features).to_csv(os.path.join(checkpoint_dir, "importance.csv"), index=False)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Train Model")
   parser.add_argument("--config_file", type=str)
   parser.add_argument("--exp_name", type=str)
   parser.add_argument("--csv_path", type=str)
   parser.add_argument("--label_col", type=str)
   parser.add_argument("--split_col", type=str)
   parser.add_argument("--model", type=str)
   parser.add_argument("--search", action="store_true")
   args = parser.parse_args()
   config = get_cfg_by_file(args.config_file)
   main()
