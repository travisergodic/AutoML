exp_dir=./checkpoints/titanic_v1/
csv_path=./data/titanic/train.csv
save_path=./predict.csv
label_col=Survived
eval_config_file=./configs/titanic/eval_cfg.py

python tools/inference.py --exp_dir ${exp_dir} \
                          --csv_path ${csv_path} \
                          --save_path ${save_path} \
                          --label_col ${label_col} \
                          --eval_config_file ${eval_config_file}

       