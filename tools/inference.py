import os
import sys
import logging
import argparse
sys.path.insert(0, os.getcwd())

import pandas as pd

from src.trainer import Predictor
from src.pipeline import build_handler_pipeline
from src.utils import get_pretty_table, load_json
from src.logger_helper import setup_logger

logger = setup_logger(level=logging.INFO)

def main():
    # data
    df = pd.read_csv(args.csv_path, low_memory=False)
    logger.info(f"Read csv file from {args.csv_path}.") 

    # data cleaning
    df=build_handler_pipeline(
        load_json(os.path.join(args.exp_dir, "handlers.json"))
    ).transform(df)

    if args.split_col:
        if args.split_col not in df:
            raise ValueError(f"{args.split_col} not in dataframe!")
        else:
            df=df[~df[args.split_col]]

    logger.info(f"Got {len(df)} data for evaluation.")

    # build predictor
    predictor=Predictor.build_from_exp_dir(args.exp_dir, args.label_col, args.metric)

    # evaluate
    if args.label_col in df:
        metric_df = pd.DataFrame.from_records([predictor.evaluate(df)])
        logger.info(get_pretty_table(metric_df, title="model performance"))
        
    # save result
    if args.save_path:
        predictor.get_pred_result(df).to_csv(args.save_path)
        logger.info(f"save prediction result at {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Prediction")
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--label_col", type=str)
    parser.add_argument("--split_col", type=str)
    parser.add_argument("--metric", type=str, nargs="+", default=[])
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    main()