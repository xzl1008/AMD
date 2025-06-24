root_path_name=./data/
seed=2024
seq_len=96
pred_len=12

# 定义一个关联数组来映射文件名到对应的model_id和data_name
declare -A file_mapping=(
    ["PEMS03.npz"]="PEMS03"
    ["PEMS04.npz"]="PEMS04"
    ["PEMS07.npz"]="PEMS07"
    ["PEMS08.npz"]="PEMS08"
)

for data_path_name in "${!file_mapping[@]}"
do
    model_id_name="${file_mapping[$data_path_name]}"
    data_name="${file_mapping[$data_path_name]}"

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
        --result_path short_result.csv
done