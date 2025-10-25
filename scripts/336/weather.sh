
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
GPU=2
root=./dataset

alpha=0.1
data_name=weather
for pred_len in 336 336 336 720 720 720
do
  export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  python -u tune1.py \
    --is_training 1 \
    --gpu $GPU \
    --root_path $root/weather/ \
    --data_path weather.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 21 \
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 1 \
    --des 'Exp' \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    --itr 1 > logs/test/new/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done
