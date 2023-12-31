# AutoML
Establish an automated, scalable, and maintainable ML pipeline to reduce the ML development cycle.

## Installation
1. **Use python version 3.10.0**
   ```
   pyenv local 3.10.0
   ```

2. **Create virtualenv & activate**
   ```
   python -m venv automl-env
   source automl-env/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

## Command
1. **Train model**
   ```bash
   python tools/train --config_file ${config_file} \
                      --exp_name ${exp_name} \
                      --csv_path ${csv_path} \
                      --label_col ${label_col} \
                      --split_col ${split_col} \
                      --test_ratio ${test_ratio} \
                      --stratify ${stratify} \
                      [--search]
   ```
   + **config_file**: Configuration file.
   + **exp_name**: Experiment name. Artifacts will be saved at `./checkpoints/{exp_name}` directory, including
     + `features.json`
     + `handlers.json`
     + `use_cols.json`
     + `importance.csv`
     + `metrics.csv`
     + `model.pkl`
     + `preprocessor.pkl`
     + `pred_result.csv`
   + **csv_path**: csv path.
   + **label_col**: Label column.
   + **split_col**: The column used for train test split.
   + **test_ratio**: Testing set ratio. Only works when `args.split_col` is `None`.
   + **stratify**: Whether to do stratify split.
   + **search**: Whether to use grid search or randomized search.

2. **Inference**
   ```bash
   python tools/inference.py --exp_dir ${exp_dir} \
                             --csv_path ${csv_path} \
                             --label_col ${label_col} \
                             --split_col ${split_col} \
                             --metric ${metric} \
                             [--save_path]
   ```
   + **exp_dir**: Experiemnt directory.
   + **csv_path**: csv_path.
   + **label_col**: Label column.
   + **split_col**: The column used for train test split. Testing set must set to `False`.
   + **metric**: Evaluation metrics.
   + **save_path**: Path to the result dataframe. If `None`, will not save the dataframe.