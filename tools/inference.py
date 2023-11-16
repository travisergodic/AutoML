import os
import logging
import argparse
import sys
sys.path.insert(0, os.getcwd())

import pandas as pd

from src.trainer import Predictor
from src.utils import get_pretty_table
from src.logger_helper import setup_logger

logger = setup_logger(level=logging.INFO)

def main():
    df = pd.read_csv(args.csv_path, low_memory=False)
    logger.info(f"Read csv file from {args.csv_path}.") 
 
    # build predictor
    predictor=Predictor.build_from_exp_dir(args.exp_dir, args.label_col, args.metric)

    # evaluate
    if args.metric:
        metric_df = pd.DataFrame.from_records([predictor.evaluate(df)])
        logger.info(get_pretty_table(metric_df, title="model performance"))
        
    # save result
    if args.save_path:
        predictor.get_pred_result(df)
        logger.info(f"save prediction result at {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Prediction")
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--time_start", type=str)
    parser.add_argument("--time_end", type=str)
    parser.add_argument("--metric", type=str, nargs="+", default=[])
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    main()