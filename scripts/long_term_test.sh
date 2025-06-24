root_path_name=./data/
seed=2024
seq_len=512

# 定义一个关联数组来映射文件名到对应的model_id和data_name
declare -A file_mapping=(
    ["weather.csv"]="weather"
    ["ETTh1.csv"]="ETTh1"
    ["ETTh2.csv"]="ETTh2"
    ["ETTm1.csv"]="ETTm1"
    ["ETTm2.csv"]="ETTm2"
    ["electricity.csv"]="electricity"
    ["exchange_rate.csv"]="exchange_rate"
    ["traffic.csv"]="traffic"
    ["solar_AL.txt"]="solar"
)

for data_path_name in "${!file_mapping[@]}"
do
    model_id_name="${file_mapping[$data_path_name]}"
    data_name="${file_mapping[$data_path_name]}"
    for pred_len in 96 192 336 720
    do
      python -u main.py \
        --seed $seed \
        --cuda "cuda:0" \
        --data $root_path_name$data_path_name \
        --feature_type M \
        --target OT \
        --checkpoint_dir ./checkpoints \
        --name $model_id_name \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --n_block 1 \
        --alpha 0.0 \
        --mix_layer_num 3 \
        --mix_layer_scale 2 \
        --patch 16 \
        --norm True \
        --layernorm True \
        --dropout 0.1 \
        --train_epochs 10 \
        --batch_size 128 \
        --learning_rate 0.00005 \
        --result_path long_result.csv
    done
done