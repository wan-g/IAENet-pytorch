export CUDA_VISIBLE_DEVICES=0

model_name=IAENet

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/MuAE-5min \
  --model_id MuAE \
  --model $model_name \
  --data DB \
  --e_layers 2 \
  --batch_size 64 \
  --d_model 2048 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --enc_in 20 \
  --static_numbers 5 \
  --loss_strategy LCRLoss\
  --prior local \
  --weight_strategy sqrt_inverse \
  ----lambda_co 0.02

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/MuAE-10min \
  --model_id MuAE \
  --model $model_name \
  --data DB \
  --e_layers 2 \
  --batch_size 64 \
  --d_model 2048 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --enc_in 20 \
  --static_numbers 5 \
  --loss_strategy LCRLoss\
  --prior local \
  --weight_strategy sqrt_inverse \
  ----lambda_co 0.02

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/MuAE-15min \
  --model_id MuAE \
  --model $model_name \
  --data DB \
  --e_layers 2 \
  --batch_size 64 \
  --d_model 2048 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --enc_in 20 \
  --static_numbers 5 \
  --loss_strategy LCRLoss\
  --prior local \
  --weight_strategy sqrt_inverse \
  ----lambda_co 0.02