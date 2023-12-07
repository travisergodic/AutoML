config_file=configs/titanic/train_cfg_v1.py
exp_name=titanic_v1
csv_path=data/titanic/train.csv
label_col=Survived
test_ratio=0.2

python tools/train.py --config_file  ${config_file} \
                      --exp_name ${exp_name} \
                      --csv_path ${csv_path} \
                      --label_col ${label_col} \
                      --test_ratio ${test_ratio}
                      # --search