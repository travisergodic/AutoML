import os
import sys
import logging
import shutil
import argparse
sys.path.insert(0, os.getcwd())

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.utils import get_cfg_by_file
from src.logger_helper import setup_logger
from src.trainer import Trainer
from src.model import MODEL
from src.metric import METRIC
from src.search import SEARCH
from src.utils import save_json
from src.pipeline import build_encoder_pipeline, build_imputer_pipeline, build_normalizer_pipeline, build_handler_pipeline

pd.options.mode.chained_assignment = None
logger = setup_logger(level=logging.INFO)


def main():
    df = pd.read_csv(args.csv_path, low_memory=False)
    logger.info(f"Read csv file from {args.csv_path}.")
    # build handler pipeline & do data cleaning 
    handler_ppipeline=build_handler_pipeline(config.list_of_handler_cfg)
    df=handler_ppipeline.fit_transform(df)

    # build preprocessor
    encoder_pipeline=build_encoder_pipeline(config.list_of_encoder_cfg)
    imputer_pipeline=build_imputer_pipeline(config.list_of_imputer_cfg)
    normalizer_pipeline=build_normalizer_pipeline(config.list_of_normalizer_cfg)
    preprocessor=Pipeline(
        [
            ("imputer", imputer_pipeline), 
            ("encoder", encoder_pipeline), 
            ("normalizer", normalizer_pipeline)
        ], verbose=True
    )

    # build model
    model=MODEL.build(**config.model_cfg)

    # build search
    if args.search and config.search_cfg:
        search = SEARCH.build(model=model, **config.search_cfg)
  
    # build metrics
    metric_dict=dict()
    for cfg in config.list_of_metric_cfg:
        calculator=METRIC.build(**cfg)
        metric_dict[str(calculator)]=calculator

    # build trainer
    trainer = Trainer(
        (model if not args.search else search),
        preprocessor, config.used_cols, args.label_col, 
        metric_dict=metric_dict
    )

    # train model
    if args.split_col:
        df_train, df_test=df[df[args.split_col]], df[~df[args.split_col]]

    elif args.test_ratio:
        df_train, df_test = train_test_split(
            df, test_size=args.test_ratio, stratify=(df[args.label_col] if args.stratify else None), random_state=42
        )
        
    else:
        df_train, df_test=df, None
        logger.info("No test set.")

    trainer.train(df_train, df_test)

    # save checkpoints
    checkpoint_dir=f"./checkpoints/{args.exp_name}"
    trainer.save(checkpoint_dir)
    save_json(config.list_of_handler_cfg, checkpoint_dir+"/handlers.json")
    logger.info(f"Save handler pipeline at {checkpoint_dir+'/handlers.json'}")
    shutil.copy(args.config_file, os.path.join(checkpoint_dir, os.path.basename(args.config_file)))
    logger.info(f"Save config file at {os.path.join(checkpoint_dir, os.path.basename(args.config_file))}")
    
    
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Train Model")
   parser.add_argument("--config_file", type=str)
   parser.add_argument("--exp_name", type=str)
   parser.add_argument("--csv_path", type=str)
   parser.add_argument("--label_col", type=str)
   parser.add_argument("--split_col", type=str)
   parser.add_argument("--test_ratio", type=float)
   parser.add_argument("--stratify", action="store_true")
   parser.add_argument("--search", action="store_true")
   args = parser.parse_args()
   config = get_cfg_by_file(args.config_file)
   main()