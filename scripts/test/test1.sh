if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/optune" ]; then
    mkdir ./logs/optune
fi

if [ ! -d "./logs/optune/tune3" ]; then
    mkdir ./logs/optune/tune3
fi

model_name=TimeBridge
seq_len=720
GPU=0
root=./dataset

alpha=0.35
data_name=ETTm2
for pred_len in 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u tune3.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --pd_layers 1 \
    --enc_in 7 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des 'Exp' \
    --n_heads 4 \
    --d_model 64  \
    --d_ff 128 \
    --lradj 'TST' \
    --period 48 \
    --train_epochs 100 \
    --learning_rate 0.0002 \
    --pct_start 0.2 \
    --patience 10 \
    --batch_size 64 \
    --alpha $alpha \
    --itr 1 | tee logs/optune/tune3/$data_name'_'$alpha'_'$model_name'_'$pred_len"__".logs
done