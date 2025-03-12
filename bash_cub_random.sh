nohup bash -c "python run_base_ft.py \
  --strategy random_blocks \
  --dataset CUB \
  --batch_size 512 \
  --num_workers 16 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-4 \
  --weight_decay 1e-4 " > cub_random_blocks.log 2>&1 &
echo "Process started with PID: $!"