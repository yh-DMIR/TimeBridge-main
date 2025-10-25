export MIOPEN_DISABLE_CACHE=1
export MIOPEN_SYSTEM_DB_PATH=""
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/test" ]; then
    mkdir ./logs/test
fi

if [ ! -d "./logs/test/new" ]; then
    mkdir ./logs/test/new
fi

model_name=TimeBridge
seq_len=96
GPU=4
root=./dataset

alpha=0.35
data_name=ETTh2
for pred_len in 336 336 336 720 720 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u tune.py \
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
    --enc_in 7 \
    --period 48 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --ca_layers 0 \
    --des 'Exp' \
    --n_heads 4 \
    --period 48 \
    --d_model 128 \
    --d_ff 128 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --patience 15 \
    --alpha $alpha \
    --batch_size 16 \
    --itr 1 > logs/test/new/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done