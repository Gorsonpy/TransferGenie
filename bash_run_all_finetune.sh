nohup bash -c "python run_all_finetune.py \
  --dataset CUB \
  --batch_size 256 \
  --num_workers 8 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --save ./checkpoints/cub_all_finetune" > cub_all_finetune.log 2>&1 &
echo "Process started with PID: $!"