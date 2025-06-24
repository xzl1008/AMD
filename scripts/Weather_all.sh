#!/bin/bash

# Define variables for your paths, model, and data
root_path_name=./data/
data_path_name=weather.csv
model_id_name=weather

seed=2024

# Define the values for seq_len and pred_len
seq_len_values=(96 192 336 512 672 720)
pred_len_values=(96 192 336 720)

# Loop over each seq_len value
for seq_len in "${seq_len_values[@]}"; do
  # Loop over each pred_len value
  for pred_len in "${pred_len_values[@]}"; do
    echo "Running with seq_len=$seq_len and pred_len=$pred_len..."

    python -u main.py \
      --seed $seed \
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
      --result_path result.csv

    echo "Finished seq_len=$seq_len and pred_len=$pred_len."
  done
done
