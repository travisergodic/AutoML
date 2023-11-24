exp_dir=./checkpoints/titanic_v1/
csv_path=./data/titanic/test.csv
save_path=./predict.csv

python tools/inference.py --exp_dir ${exp_dir} \
                          --csv_path ${csv_path} \
                          --save_path ${save_path}

       