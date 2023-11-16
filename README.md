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
                      --model ${model} \
                      [--search]
   ```
   + **config_file**: Configuration file.
   + **exp_name**: Experiment name. All artifacts will be saved at `./checkpoints/{exp_name}` directory.
   + **csv_path**: csv path.
   + **model**: Model name.
   + **search**: Whether to use grid search or randomized search.
   + **label_col**: Label column.
   + **split_col**: The column used for train test split.


2. **Inference**
   ```bash
   python tools/inference.py --exp_dir ${exp_dir} \
                             --csv_path ${csv_path} \
                             --label_col ${label_col} \
                             --
                             --metric ${metric} \
                             [--save_path]
   ```
   + **exp_dir**: Experiemnt directory.
   + **csv_path**: csv_path.
   + **label_col**: Label column.
   + **split_col**: The column used for train test split.
   + **metric**: Evaluation metrics.
   + **save_path**: Path to the result dataframe. If `None`, will not save the dataframe.